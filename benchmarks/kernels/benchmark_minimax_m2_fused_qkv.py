#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MiniMax-M2 fused QKV (CUTLASS epilogue + norm + RoPE).

Compares:
  - fused (scheme C): CUTLASS epilogue fused sum + comm + finalize
  - staged (scheme B): GEMM + standalone sum_qk kernel + comm + finalize
  - eager: PyTorch linear + forward_qk + rotary baseline

Single GPU:
  python benchmarks/kernels/benchmark_minimax_m2_fused_qkv.py

8-GPU tensor parallel (real NCCL all_reduce on qk_sum_sq):
  bash benchmarks/kernels/run_minimax_m2_fused_qkv_8gpu.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_VLLM_ROOT = Path(__file__).resolve().parents[2]
if str(_VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLLM_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F

import vllm._C  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.minimax_m2_fused_qkv import (
    minimax_m2_fused_qkv_allocate_workspace,
    minimax_m2_fused_qkv_available,
    minimax_m2_fused_qkv_communicate,
    minimax_m2_fused_qkv_compute,
    minimax_m2_fused_qkv_epilogue_sum_sq,
    minimax_m2_fused_qkv_finalize,
    minimax_m2_fused_qkv_from_hidden,
    minimax_m2_fused_qkv_gemm_only,
    minimax_m2_fused_qkv_staged_benchmark_available,
)
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

try:
    import triton.testing as triton_testing
except ImportError:  # pragma: no cover
    triton_testing = None


# MiniMax-M2 defaults (https://huggingface.co/MiniMaxAI/MiniMax-M2)
MINIMAX_M2_HIDDEN_SIZE = 3072
MINIMAX_M2_NUM_HEADS_Q = 48
MINIMAX_M2_NUM_HEADS_KV = 8
MINIMAX_M2_HEAD_DIM = 128
MINIMAX_M2_RMS_EPS = 1e-6


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    is_distributed: bool


@dataclass(frozen=True)
class BenchConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_k: int
    head_dim: int
    tp_world: int
    eps: float
    dtype: torch.dtype

    @property
    def q_size(self) -> int:
        return self.num_heads_q * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_heads_k * self.head_dim

    @property
    def out_dim(self) -> int:
        return self.q_size + 2 * self.kv_size


@dataclass
class BenchTensors:
    hidden: torch.Tensor
    weight: torch.Tensor
    qkv: torch.Tensor
    positions: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    cos_sin_cache: torch.Tensor
    qk_sum_sq: torch.Tensor
    rope: RotaryEmbedding
    q_norm_module: MiniMaxText01RMSNormTP | None = None
    k_norm_module: MiniMaxText01RMSNormTP | None = None


def init_dist(device_arg: str) -> DistContext:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        if get_tensor_model_parallel_world_size() != world_size:
            raise RuntimeError(
                f"TP world size mismatch: expected {world_size}, "
                f"got {get_tensor_model_parallel_world_size()}"
            )
        if get_tensor_model_parallel_rank() != rank:
            raise RuntimeError(
                f"TP rank mismatch: expected {rank}, "
                f"got {get_tensor_model_parallel_rank()}"
            )
        return DistContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            is_distributed=True,
        )

    device = torch.device(device_arg)
    if device.type != "cuda":
        raise SystemExit(f"CUDA device required, got {device_arg}")
    torch.cuda.set_device(device)
    return DistContext(
        rank=0,
        world_size=1,
        local_rank=device.index or 0,
        device=device,
        is_distributed=False,
    )


def make_minimax_m2_config(tp_world: int) -> BenchConfig:
    if MINIMAX_M2_NUM_HEADS_Q % tp_world != 0:
        raise ValueError(
            f"num_heads_q={MINIMAX_M2_NUM_HEADS_Q} not divisible by tp_world={tp_world}"
        )
    if MINIMAX_M2_NUM_HEADS_KV % tp_world != 0:
        raise ValueError(
            f"num_heads_kv={MINIMAX_M2_NUM_HEADS_KV} not divisible by tp_world={tp_world}"
        )
    return BenchConfig(
        hidden_size=MINIMAX_M2_HIDDEN_SIZE,
        num_heads_q=MINIMAX_M2_NUM_HEADS_Q // tp_world,
        num_heads_k=MINIMAX_M2_NUM_HEADS_KV // tp_world,
        head_dim=MINIMAX_M2_HEAD_DIM,
        tp_world=tp_world,
        eps=MINIMAX_M2_RMS_EPS,
        dtype=torch.bfloat16,
    )


def make_tensors(cfg: BenchConfig, num_tokens: int, dist_ctx: DistContext) -> BenchTensors:
    device = dist_ctx.device
    hidden = torch.randn(
        num_tokens, cfg.hidden_size, dtype=cfg.dtype, device=device
    )
    weight = torch.randn(cfg.out_dim, cfg.hidden_size, dtype=cfg.dtype, device=device)
    qkv = torch.empty(num_tokens, cfg.out_dim, dtype=cfg.dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm_weight = torch.randn(cfg.q_size, dtype=cfg.dtype, device=device)
    k_norm_weight = torch.randn(cfg.kv_size, dtype=cfg.dtype, device=device)

    rope = RotaryEmbedding(
        head_size=cfg.head_dim,
        rotary_dim=cfg.head_dim,
        max_position_embeddings=8192,
        base=10000.0,
        is_neox_style=True,
        dtype=cfg.dtype,
    ).to(device)

    qk_sum_sq = minimax_m2_fused_qkv_allocate_workspace(num_tokens, device)

    q_norm_module = None
    k_norm_module = None
    if dist_ctx.is_distributed and cfg.tp_world > 1:
        q_norm_module = MiniMaxText01RMSNormTP(
            MINIMAX_M2_NUM_HEADS_Q * cfg.head_dim, eps=cfg.eps
        ).to(device)
        k_norm_module = MiniMaxText01RMSNormTP(
            MINIMAX_M2_NUM_HEADS_KV * cfg.head_dim, eps=cfg.eps
        ).to(device)
        q_norm_module.weight.data.copy_(q_norm_weight)
        k_norm_module.weight.data.copy_(k_norm_weight)

    return BenchTensors(
        hidden=hidden,
        weight=weight,
        qkv=qkv,
        positions=positions,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=rope.cos_sin_cache,
        qk_sum_sq=qk_sum_sq,
        rope=rope,
        q_norm_module=q_norm_module,
        k_norm_module=k_norm_module,
    )


def _clone_inputs(t: BenchTensors) -> BenchTensors:
    return BenchTensors(
        hidden=t.hidden.clone(),
        weight=t.weight,
        qkv=t.qkv.clone(),
        positions=t.positions,
        q_norm_weight=t.q_norm_weight,
        k_norm_weight=t.k_norm_weight,
        cos_sin_cache=t.cos_sin_cache,
        qk_sum_sq=t.qk_sum_sq.clone().zero_(),
        rope=t.rope,
        q_norm_module=t.q_norm_module,
        k_norm_module=t.k_norm_module,
    )


def run_fused_full(t: BenchTensors, cfg: BenchConfig) -> None:
    minimax_m2_fused_qkv_from_hidden(
        hidden_states=t.hidden,
        positions=t.positions,
        qkv=t.qkv,
        qkv_weight=t.weight,
        qkv_bias=None,
        q_norm_weight=t.q_norm_weight,
        k_norm_weight=t.k_norm_weight,
        cos_sin_cache=t.cos_sin_cache,
        num_heads_q=cfg.num_heads_q,
        num_heads_k=cfg.num_heads_k,
        num_heads_v=cfg.num_heads_k,
        head_dim=cfg.head_dim,
        eps=cfg.eps,
        is_neox=True,
        tp_world=cfg.tp_world,
    )


def _compute_kwargs(
    t: BenchTensors, cfg: BenchConfig, *, use_standalone_sum_epilogue: bool = False
) -> dict:
    return dict(
        hidden_states=t.hidden,
        positions=t.positions,
        qkv=t.qkv,
        qkv_weight=t.weight,
        qkv_bias=None,
        q_norm_weight=t.q_norm_weight,
        k_norm_weight=t.k_norm_weight,
        cos_sin_cache=t.cos_sin_cache,
        qk_sum_sq=t.qk_sum_sq,
        num_heads_q=cfg.num_heads_q,
        num_heads_k=cfg.num_heads_k,
        num_heads_v=cfg.num_heads_k,
        head_dim=cfg.head_dim,
        eps=cfg.eps,
        is_neox=True,
        tp_world=cfg.tp_world,
        use_standalone_sum_epilogue=use_standalone_sum_epilogue,
    )


def run_fused_compute(t: BenchTensors, cfg: BenchConfig) -> None:
    t.qk_sum_sq.zero_()
    minimax_m2_fused_qkv_compute(**_compute_kwargs(t, cfg))


def run_staged_compute(t: BenchTensors, cfg: BenchConfig) -> None:
    t.qk_sum_sq.zero_()
    minimax_m2_fused_qkv_compute(
        **_compute_kwargs(t, cfg, use_standalone_sum_epilogue=True)
    )


def run_staged_gemm(t: BenchTensors, cfg: BenchConfig) -> None:
    minimax_m2_fused_qkv_gemm_only(t.hidden, t.qkv, t.weight, qkv_bias=None)


def run_staged_sum(t: BenchTensors, cfg: BenchConfig) -> None:
    t.qk_sum_sq.zero_()
    minimax_m2_fused_qkv_epilogue_sum_sq(
        t.qkv, t.qk_sum_sq, cfg.q_size, cfg.kv_size, qkv_bias=None
    )


def run_staged_full(t: BenchTensors, cfg: BenchConfig) -> None:
    run_staged_compute(t, cfg)
    run_fused_communicate(t, cfg)
    run_fused_finalize(t, cfg)


def run_fused_communicate(t: BenchTensors, cfg: BenchConfig) -> None:
    minimax_m2_fused_qkv_communicate(t.qk_sum_sq, cfg.tp_world)


def run_fused_finalize(t: BenchTensors, cfg: BenchConfig) -> None:
    minimax_m2_fused_qkv_finalize(
        positions=t.positions,
        qkv=t.qkv,
        q_norm_weight=t.q_norm_weight,
        k_norm_weight=t.k_norm_weight,
        cos_sin_cache=t.cos_sin_cache,
        qk_sum_sq=t.qk_sum_sq,
        num_heads_q=cfg.num_heads_q,
        num_heads_k=cfg.num_heads_k,
        num_heads_v=cfg.num_heads_k,
        head_dim=cfg.head_dim,
        eps=cfg.eps,
        is_neox=True,
        tp_world=cfg.tp_world,
    )


def run_eager_tp1(t: BenchTensors, cfg: BenchConfig) -> None:
    qkv = F.linear(t.hidden, t.weight, None)
    q, k, v = qkv.split([cfg.q_size, cfg.kv_size, cfg.kv_size], dim=-1)

    class _NormStub:
        tp_world = 1
        variance_epsilon = cfg.eps

    q_norm = _NormStub()
    q_norm.weight = t.q_norm_weight
    k_norm = _NormStub()
    k_norm.weight = t.k_norm_weight

    q, k = MiniMaxText01RMSNormTP.forward_qk(
        q_norm, k_norm, q.contiguous(), k.contiguous()
    )
    q, k = t.rope.forward_native(t.positions, q, k)
    t.qkv.copy_(torch.cat([q, k, v], dim=-1))


def run_eager_tp(t: BenchTensors, cfg: BenchConfig) -> None:
    assert t.q_norm_module is not None and t.k_norm_module is not None
    qkv = F.linear(t.hidden, t.weight, None)
    q, k, v = qkv.split([cfg.q_size, cfg.kv_size, cfg.kv_size], dim=-1)
    q, k = MiniMaxText01RMSNormTP.forward_qk(
        t.q_norm_module, t.k_norm_module, q.contiguous(), k.contiguous()
    )
    q, k = t.rope.forward_native(t.positions, q, k)
    t.qkv.copy_(torch.cat([q, k, v], dim=-1))


def run_eager_gemm_only(t: BenchTensors, cfg: BenchConfig) -> None:
    t.qkv.copy_(F.linear(t.hidden, t.weight, None))


def _sync_dist(dist_ctx: DistContext) -> None:
    if dist_ctx.is_distributed:
        dist.barrier()
    else:
        torch.cuda.synchronize()


def _reduce_max_us(local_us: float, dist_ctx: DistContext) -> float:
    if not dist_ctx.is_distributed:
        return local_us
    t = torch.tensor([local_us], device=dist_ctx.device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def bench_fn(fn, warmup: int, iters: int, dist_ctx: DistContext) -> float:
    """Return median latency in microseconds (max across TP ranks when distributed)."""
    for _ in range(warmup):
        fn()
    _sync_dist(dist_ctx)

    if triton_testing is not None and not dist_ctx.is_distributed:
        ms, _, _ = triton_testing.do_bench(
            fn, warmup=0, rep=iters, quantiles=[0.5, 0.2, 0.8]
        )
        return ms * 1e3

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us: list[float] = []
    for _ in range(iters):
        _sync_dist(dist_ctx)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1e3)
    times_us.sort()
    median_us = times_us[len(times_us) // 2]
    return _reduce_max_us(median_us, dist_ctx)


def format_row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))


def log_rank0(dist_ctx: DistContext, msg: str) -> None:
    if dist_ctx.rank == 0:
        print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MiniMax-M2 fused QKV")
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 1024, 4096],
        help="Token counts to benchmark",
    )
    parser.add_argument(
        "--tp-world",
        type=int,
        default=None,
        help="TP size (default: WORLD_SIZE under torchrun, else 1)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Break fused path into compute / communicate / finalize",
    )
    parser.add_argument(
        "--include-gemm-only",
        action="store_true",
        help="Also benchmark raw qkv_proj GEMM (eager linear only)",
    )
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="Also benchmark eager linear+norm+rope baseline",
    )
    parser.add_argument(
        "--compare-staged",
        action="store_true",
        help="Also benchmark scheme B: GEMM + standalone sum_qk + comm + finalize",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if not current_platform.is_cuda_alike():
        raise SystemExit("CUDA is required")
    if not minimax_m2_fused_qkv_available():
        raise SystemExit(
            "minimax_m2_fused_qkv CUDA ops not found; build vLLM with minimax_m2_qkv.cu"
        )

    # vLLM distributed init and CustomOp modules require an active config context.
    with set_current_vllm_config(VllmConfig()):
        _run_benchmark(args)


def _run_benchmark(args: argparse.Namespace) -> None:
    dist_ctx = init_dist(args.device)

    if dist_ctx.is_distributed:
        tp_world = args.tp_world if args.tp_world is not None else dist_ctx.world_size
        if tp_world != dist_ctx.world_size:
            raise SystemExit(
                f"--tp-world={tp_world} must match torchrun WORLD_SIZE="
                f"{dist_ctx.world_size}"
            )
    else:
        tp_world = args.tp_world if args.tp_world is not None else 1
        if tp_world > 1:
            raise SystemExit(
                f"--tp-world={tp_world} requires multi-GPU launch via torchrun, e.g.\n"
                "  torchrun --nproc_per_node=8 "
                "benchmarks/kernels/benchmark_minimax_m2_fused_qkv.py --profile-stages"
            )

    cfg = make_minimax_m2_config(tp_world)
    compare_eager = args.compare_eager or tp_world == 1
    compare_staged = args.compare_staged or tp_world > 1
    if compare_staged and tp_world <= 1:
        log_rank0(dist_ctx, "  note: --compare-staged is only meaningful for tp_world > 1")
        compare_staged = False
    if compare_staged and not minimax_m2_fused_qkv_staged_benchmark_available():
        log_rank0(
            dist_ctx,
            "  note: scheme-B (staged) benchmark needs recompile; skipping staged columns. "
            "Run: pip install -e .",
        )
        compare_staged = False

    log_rank0(dist_ctx, "MiniMax-M2 fused QKV benchmark")
    log_rank0(
        dist_ctx,
        f"  rank={dist_ctx.rank}/{dist_ctx.world_size}  device={dist_ctx.device}  "
        f"dtype={cfg.dtype}  tp_world={cfg.tp_world}",
    )
    log_rank0(
        dist_ctx,
        f"  hidden={cfg.hidden_size}  local_q_heads={cfg.num_heads_q}  "
        f"local_kv_heads={cfg.num_heads_k}  head_dim={cfg.head_dim}",
    )
    log_rank0(
        dist_ctx,
        f"  local qkv out dim={cfg.out_dim}  warmup={args.warmup}  iters={args.iters}",
    )
    if dist_ctx.is_distributed:
        log_rank0(
            dist_ctx,
            "  distributed: real NCCL all_reduce on qk_sum_sq; latency = max(rank)",
        )
    log_rank0(dist_ctx, "")

    headers = ["num_tokens", "fused_us"]
    if compare_staged:
        headers.append("staged_us")
    if args.profile_stages:
        headers.extend(["compute_us", "comm_us", "finalize_us"])
        if compare_staged:
            headers.extend(["gemm_us", "sum_us"])
    if compare_eager:
        headers.append("eager_us")
        headers.append("fused_speedup")
        if compare_staged:
            headers.append("staged_speedup")
    if args.include_gemm_only:
        headers.append("gemm_only_us")
    widths = [max(len(h), 12) for h in headers]

    rows: list[list[str]] = []

    for num_tokens in args.num_tokens:
        base = make_tensors(cfg, num_tokens, dist_ctx)

        run_fused_full(_clone_inputs(base), cfg)
        _sync_dist(dist_ctx)

        fused_us = bench_fn(
            lambda: run_fused_full(_clone_inputs(base), cfg),
            args.warmup,
            args.iters,
            dist_ctx,
        )

        row: list[str] = [str(num_tokens), f"{fused_us:.2f}"]

        if compare_staged:
            staged_us = bench_fn(
                lambda: run_staged_full(_clone_inputs(base), cfg),
                args.warmup,
                args.iters,
                dist_ctx,
            )
            row.append(f"{staged_us:.2f}")

        if args.profile_stages:
            stage = _clone_inputs(base)
            run_fused_compute(stage, cfg)
            _sync_dist(dist_ctx)

            compute_us = bench_fn(
                lambda: run_fused_compute(_clone_inputs(base), cfg),
                args.warmup,
                args.iters,
                dist_ctx,
            )

            # communicate: all ranks must enter together each iteration
            def _bench_comm() -> None:
                t = _clone_inputs(base)
                # distinct local partial sums per rank
                t.qk_sum_sq.fill_(float(dist_ctx.rank + 1))
                run_fused_communicate(t, cfg)

            comm_us = bench_fn(_bench_comm, args.warmup, args.iters, dist_ctx)

            # finalize: use post-allreduce qk_sum_sq
            fin_stage = _clone_inputs(base)
            run_fused_compute(fin_stage, cfg)
            run_fused_communicate(fin_stage, cfg)
            _sync_dist(dist_ctx)

            finalize_us = bench_fn(
                lambda: run_fused_finalize(fin_stage, cfg),
                args.warmup,
                args.iters,
                dist_ctx,
            )
            row.extend(
                [f"{compute_us:.2f}", f"{comm_us:.2f}", f"{finalize_us:.2f}"]
            )

            if compare_staged:
                gemm_us = bench_fn(
                    lambda: run_staged_gemm(_clone_inputs(base), cfg),
                    args.warmup,
                    args.iters,
                    dist_ctx,
                )

                gemm_prefill = _clone_inputs(base)
                run_staged_gemm(gemm_prefill, cfg)
                _sync_dist(dist_ctx)

                def _bench_sum_only() -> None:
                    t = _clone_inputs(base)
                    t.qkv.copy_(gemm_prefill.qkv)
                    run_staged_sum(t, cfg)

                sum_us = bench_fn(_bench_sum_only, args.warmup, args.iters, dist_ctx)
                row.extend([f"{gemm_us:.2f}", f"{sum_us:.2f}"])

        if compare_eager:
            if tp_world == 1:
                eager_fn = lambda: run_eager_tp1(_clone_inputs(base), cfg)
            else:
                eager_fn = lambda: run_eager_tp(_clone_inputs(base), cfg)
            eager_us = bench_fn(eager_fn, args.warmup, args.iters, dist_ctx)
            fused_speedup = eager_us / fused_us if fused_us > 0 else float("inf")
            row.append(f"{eager_us:.2f}")
            row.append(f"{fused_speedup:.2f}x")
            if compare_staged:
                staged_speedup = (
                    eager_us / staged_us if staged_us > 0 else float("inf")
                )
                row.append(f"{staged_speedup:.2f}x")

        if args.include_gemm_only:
            gemm_us = bench_fn(
                lambda: run_eager_gemm_only(_clone_inputs(base), cfg),
                args.warmup,
                args.iters,
                dist_ctx,
            )
            row.append(f"{gemm_us:.2f}")

        rows.append(row)

    if dist_ctx.rank == 0:
        print(format_row(headers, widths))
        print(format_row(["-" * w for w in widths], widths))
        for row in rows:
            print(format_row(row, widths))
        print()
        print("fused_us   = scheme C: CUTLASS epilogue fused sum + comm + finalize")
        if compare_staged:
            print("staged_us  = scheme B: GEMM + standalone sum_qk + comm + finalize")
        if args.profile_stages:
            print("  compute  = fused scheme-C compute (GEMM + epilogue sum)")
            print("  comm     = all_reduce(qk_sum_sq) over TP group")
            print("  finalize = norm + RoPE")
            if compare_staged:
                print("  gemm_us  = staged local GEMM only (CUTLASS TrivialEpilogue)")
                print("  sum_us   = staged standalone sum_qk kernel (after GEMM)")
        if compare_eager:
            print("eager_us   = linear + forward_qk + rotary (PyTorch baseline)")
            print("fused_speedup  = eager_us / fused_us")
            if compare_staged:
                print("staged_speedup = eager_us / staged_us")

    if dist_ctx.is_distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
