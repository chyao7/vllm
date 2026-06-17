#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Accuracy test: MiniMax-M2 fused QKV.

Default (--mode fp8): block FP8 W8A8 weights (128x128), matching production
MiniMax-M2 + Fp8LinearMethod.  TP>1 uses standalone sum epilogue (scheme B).

Single GPU:
  python benchmarks/kernels/test_minimax_m2_fused_qkv_accuracy.py

8-GPU TP:
  python -m torch.distributed.run --standalone --nproc_per_node=8 \\
    benchmarks/kernels/test_minimax_m2_fused_qkv_accuracy.py

BF16-only (incl. scheme C vs B):
  python benchmarks/kernels/test_minimax_m2_fused_qkv_accuracy.py --mode bf16
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
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.model_executor.layers.minimax_m2_fused_qkv import (
    minimax_m2_fused_qkv_allocate_workspace,
    minimax_m2_fused_qkv_available,
    minimax_m2_fused_qkv_communicate,
    minimax_m2_fused_qkv_compute,
    minimax_m2_fused_qkv_finalize,
    minimax_m2_fused_qkv_fp8_post_available,
    minimax_m2_fused_qkv_from_hidden,
    minimax_m2_fused_qkv_staged_benchmark_available,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import per_block_cast_to_fp8

MINIMAX_M2_HIDDEN_SIZE = 3072
MINIMAX_M2_NUM_HEADS_Q = 48
MINIMAX_M2_NUM_HEADS_KV = 8
MINIMAX_M2_HEAD_DIM = 128
MINIMAX_M2_RMS_EPS = 1e-6
MINIMAX_M2_BLOCK_SIZE = [128, 128]

# bf16 activations + post-GEMM kernels
ATOL_TP1 = 0.04
RTOL_TP1 = 0.02
# TP finalize CUDA norm+RoPE vs forward_native (~7 on q/k; ~10.5 FP8 large-M)
ATOL_TP8_OUT = 11.0
RTOL_TP8_OUT = 0.05
SUM_SQ_ATOL = 10.0
SUM_SQ_RTOL = 1e-3
# FP8 W8A8 GEMM adds extra quantization noise on q/k
ATOL_FP8_EXTRA = 0.15
RTOL_FP8_EXTRA = 0.15
# Scheme C (bf16 only) must match B
ATOL_C_VS_B = 0.04
RTOL_C_VS_B = 0.02

TOKEN_COUNTS = [1, 4, 16, 64, 256, 1024, 4096]


@dataclass(frozen=True)
class DistCtx:
    rank: int
    world_size: int
    device: torch.device
    is_distributed: bool


@dataclass(frozen=True)
class Cfg:
    hidden_size: int
    num_heads_q: int
    num_heads_k: int
    head_dim: int
    tp_world: int
    eps: float
    act_dtype: torch.dtype
    use_fp8: bool

    @property
    def q_size(self) -> int:
        return self.num_heads_q * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_heads_k * self.head_dim

    @property
    def out_dim(self) -> int:
        return self.q_size + 2 * self.kv_size


@dataclass(frozen=True)
class WeightBundle:
    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    fp8_block_linear: W8A8BlockFp8LinearOp | None


def init_dist() -> DistCtx:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        return DistCtx(
            rank=rank, world_size=world_size, device=device, is_distributed=True
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return DistCtx(rank=0, world_size=1, device=device, is_distributed=False)


def make_cfg(tp_world: int, use_fp8: bool) -> Cfg:
    return Cfg(
        hidden_size=MINIMAX_M2_HIDDEN_SIZE,
        num_heads_q=MINIMAX_M2_NUM_HEADS_Q // tp_world,
        num_heads_k=MINIMAX_M2_NUM_HEADS_KV // tp_world,
        head_dim=MINIMAX_M2_HEAD_DIM,
        tp_world=tp_world,
        eps=MINIMAX_M2_RMS_EPS,
        act_dtype=torch.bfloat16,
        use_fp8=use_fp8,
    )


def make_fp8_block_linear() -> W8A8BlockFp8LinearOp:
    block_m, block_k = MINIMAX_M2_BLOCK_SIZE
    return W8A8BlockFp8LinearOp(
        weight_group_shape=GroupShape(block_m, block_k),
        act_quant_group_shape=GroupShape(1, block_k),
        cutlass_block_fp8_supported=CUTLASS_BLOCK_FP8_SUPPORTED,
    )


def log0(ctx: DistCtx, msg: str) -> None:
    if ctx.rank == 0:
        print(msg, flush=True)


def _quantize_block_fp8(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w_fp8, w_scale = per_block_cast_to_fp8(
        weight_bf16.float(), MINIMAX_M2_BLOCK_SIZE, use_ue8m0=False
    )
    return process_fp8_weight_block_strategy(w_fp8, w_scale)


def _run_qkv_gemm(
    hidden: torch.Tensor,
    bundle: WeightBundle,
) -> torch.Tensor:
    if bundle.fp8_block_linear is not None:
        assert bundle.weight_scale is not None
        return bundle.fp8_block_linear.apply(
            hidden, bundle.weight, bundle.weight_scale, bias=None
        )
    return F.linear(hidden, bundle.weight, None)


def _reference_qk_sum_sq(qkv: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    q, k, _v = qkv.split([cfg.q_size, cfg.kv_size, cfg.kv_size], dim=-1)
    sq_q = q.float().pow(2).sum(dim=-1)
    sq_k = k.float().pow(2).sum(dim=-1)
    return torch.stack([sq_q, sq_k], dim=-1)


def _apply_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope: RotaryEmbedding,
    cfg: Cfg,
    q_norm_module: MiniMaxText01RMSNormTP | None,
    k_norm_module: MiniMaxText01RMSNormTP | None,
) -> torch.Tensor:
    q, k, v = qkv.split([cfg.q_size, cfg.kv_size, cfg.kv_size], dim=-1)
    if cfg.tp_world > 1:
        assert q_norm_module is not None and k_norm_module is not None
        q, k = MiniMaxText01RMSNormTP.forward_qk(
            q_norm_module, k_norm_module, q.contiguous(), k.contiguous()
        )
    else:
        orig_dtype = q.dtype
        q = q.float()
        k = k.float()
        q_var = q.pow(2).mean(dim=-1, keepdim=True)
        k_var = k.pow(2).mean(dim=-1, keepdim=True)
        q = q * torch.rsqrt(q_var + cfg.eps) * q_norm_weight.float()
        k = k * torch.rsqrt(k_var + cfg.eps) * k_norm_weight.float()
        q = q.to(orig_dtype)
        k = k.to(orig_dtype)
    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


def run_eager(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    bundle: WeightBundle,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope: RotaryEmbedding,
    cfg: Cfg,
    q_norm_module: MiniMaxText01RMSNormTP | None,
    k_norm_module: MiniMaxText01RMSNormTP | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    qkv_pre = _run_qkv_gemm(hidden, bundle)
    out = _apply_norm_rope(
        qkv_pre, positions, q_norm_weight, k_norm_weight, rope, cfg,
        q_norm_module, k_norm_module,
    )
    return out, _reference_qk_sum_sq(qkv_pre, cfg)


def _fused_fp8_kwargs(bundle: WeightBundle) -> dict:
    if bundle.fp8_block_linear is None:
        return {}
    return {
        "qkv_weight_scale": bundle.weight_scale,
        "fp8_block_linear": bundle.fp8_block_linear,
    }


def run_fused_from_hidden(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    bundle: WeightBundle,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cfg: Cfg,
    *,
    use_standalone_sum_epilogue: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    qkv = torch.empty(
        hidden.size(0), cfg.out_dim, dtype=cfg.act_dtype, device=hidden.device
    )
    qk_sum_sq = minimax_m2_fused_qkv_allocate_workspace(hidden.size(0), hidden.device)
    local_sum_sq = None

    if cfg.tp_world > 1 or not cfg.use_fp8:
        minimax_m2_fused_qkv_compute(
            hidden_states=hidden,
            positions=positions,
            qkv=qkv,
            qkv_weight=bundle.weight,
            qkv_bias=None,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            qk_sum_sq=qk_sum_sq,
            num_heads_q=cfg.num_heads_q,
            num_heads_k=cfg.num_heads_k,
            num_heads_v=cfg.num_heads_k,
            head_dim=cfg.head_dim,
            eps=cfg.eps,
            is_neox=True,
            tp_world=cfg.tp_world,
            use_standalone_sum_epilogue=use_standalone_sum_epilogue,
            **_fused_fp8_kwargs(bundle),
        )
        if cfg.tp_world > 1:
            local_sum_sq = qk_sum_sq.clone()
            minimax_m2_fused_qkv_communicate(qk_sum_sq, cfg.tp_world)
            minimax_m2_fused_qkv_finalize(
                positions=positions,
                qkv=qkv,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                cos_sin_cache=cos_sin_cache,
                qk_sum_sq=qk_sum_sq,
                num_heads_q=cfg.num_heads_q,
                num_heads_k=cfg.num_heads_k,
                num_heads_v=cfg.num_heads_k,
                head_dim=cfg.head_dim,
                eps=cfg.eps,
                is_neox=True,
                tp_world=cfg.tp_world,
            )
    else:
        minimax_m2_fused_qkv_from_hidden(
            hidden_states=hidden,
            positions=positions,
            qkv=qkv,
            qkv_weight=bundle.weight,
            qkv_bias=None,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            num_heads_q=cfg.num_heads_q,
            num_heads_k=cfg.num_heads_k,
            num_heads_v=cfg.num_heads_k,
            head_dim=cfg.head_dim,
            eps=cfg.eps,
            is_neox=True,
            tp_world=cfg.tp_world,
            **_fused_fp8_kwargs(bundle),
        )

    return qkv, local_sum_sq


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _check_close(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    atol: float,
    rtol: float,
    ctx: DistCtx,
) -> bool:
    max_diff = _max_diff(got, ref)
    try:
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
        status = "PASS"
        detail = ""
    except AssertionError as e:
        status = "FAIL"
        detail = str(e).split("\n")[0]
    if ctx.rank == 0:
        print(
            f"  {name:14s} {status:4s}  max_diff={max_diff:.6e}  {detail}",
            flush=True,
        )
    return status == "PASS"


def _make_tensors(
    num_tokens: int, cfg: Cfg, ctx: DistCtx, seed: int, fp8_linear: W8A8BlockFp8LinearOp | None
):
    rank = get_tensor_model_parallel_rank() if ctx.is_distributed else 0
    global_q_size = MINIMAX_M2_NUM_HEADS_Q * cfg.head_dim
    global_kv_size = MINIMAX_M2_NUM_HEADS_KV * cfg.head_dim
    global_out_dim = global_q_size + 2 * global_kv_size

    if ctx.rank == 0:
        gen = torch.Generator(device=ctx.device)
        gen.manual_seed(seed)
        hidden = torch.randn(
            num_tokens, cfg.hidden_size, dtype=cfg.act_dtype, device=ctx.device, generator=gen
        )
        weight_full = torch.randn(
            global_out_dim, cfg.hidden_size, dtype=cfg.act_dtype, device=ctx.device, generator=gen
        )
        q_norm_full = torch.ones(global_q_size, dtype=cfg.act_dtype, device=ctx.device)
        k_norm_full = torch.ones(global_kv_size, dtype=cfg.act_dtype, device=ctx.device)
        q_norm_full = q_norm_full.normal_(mean=1.0, std=0.1, generator=gen)
        k_norm_full = k_norm_full.normal_(mean=1.0, std=0.1, generator=gen)
    else:
        hidden = torch.empty(
            num_tokens, cfg.hidden_size, dtype=cfg.act_dtype, device=ctx.device
        )
        weight_full = torch.empty(
            global_out_dim, cfg.hidden_size, dtype=cfg.act_dtype, device=ctx.device
        )
        q_norm_full = torch.empty(global_q_size, dtype=cfg.act_dtype, device=ctx.device)
        k_norm_full = torch.empty(global_kv_size, dtype=cfg.act_dtype, device=ctx.device)

    if ctx.is_distributed:
        dist.broadcast(hidden, src=0)
        dist.broadcast(weight_full, src=0)
        dist.broadcast(q_norm_full, src=0)
        dist.broadcast(k_norm_full, src=0)

    weight_shard = weight_full[rank * cfg.out_dim : (rank + 1) * cfg.out_dim]
    q_norm_weight = q_norm_full[rank * cfg.q_size : (rank + 1) * cfg.q_size]
    k_norm_weight = k_norm_full[rank * cfg.kv_size : (rank + 1) * cfg.kv_size]

    if cfg.use_fp8:
        weight_fp8, weight_scale = _quantize_block_fp8(weight_shard)
        bundle = WeightBundle(weight_fp8, weight_scale, fp8_linear)
    else:
        bundle = WeightBundle(weight_shard, None, None)

    positions = torch.arange(num_tokens, dtype=torch.long, device=ctx.device)
    rope = RotaryEmbedding(
        head_size=cfg.head_dim,
        rotary_dim=cfg.head_dim,
        max_position_embeddings=8192,
        base=10000.0,
        is_neox_style=True,
        dtype=cfg.act_dtype,
    ).to(ctx.device)

    q_norm_module = None
    k_norm_module = None
    if cfg.tp_world > 1:
        q_norm_module = MiniMaxText01RMSNormTP(global_q_size, eps=cfg.eps).to(ctx.device)
        k_norm_module = MiniMaxText01RMSNormTP(global_kv_size, eps=cfg.eps).to(ctx.device)
        q_norm_module.weight.data.copy_(q_norm_weight)
        k_norm_module.weight.data.copy_(k_norm_weight)

    return hidden, bundle, positions, q_norm_weight, k_norm_weight, rope, q_norm_module, k_norm_module


def _out_tolerance(cfg: Cfg) -> tuple[float, float]:
    if cfg.tp_world > 1:
        atol, rtol = ATOL_TP8_OUT, RTOL_TP8_OUT
    else:
        atol, rtol = ATOL_TP1, RTOL_TP1
    if cfg.use_fp8:
        atol += ATOL_FP8_EXTRA
        rtol += RTOL_FP8_EXTRA
    return atol, rtol


def run_accuracy(ctx: DistCtx, mode: str) -> None:
    if not minimax_m2_fused_qkv_available():
        raise SystemExit("minimax_m2_fused_qkv ops not available")

    use_fp8 = mode == "fp8"
    if use_fp8 and not minimax_m2_fused_qkv_fp8_post_available():
        raise SystemExit(
            "FP8 mode requires minimax_m2_fused_qkv_post_fused; rebuild vLLM"
        )

    tp_world = ctx.world_size if ctx.is_distributed else 1
    cfg = make_cfg(tp_world, use_fp8=use_fp8)
    has_staged = minimax_m2_fused_qkv_staged_benchmark_available() and not use_fp8
    fp8_linear = make_fp8_block_linear() if use_fp8 else None
    out_atol, out_rtol = _out_tolerance(cfg)

    log0(ctx, "MiniMax-M2 fused QKV accuracy test")
    log0(ctx, f"  mode={'FP8 block W8A8' if use_fp8 else 'BF16'}  tp_world={cfg.tp_world}")
    log0(ctx, f"  act_dtype={cfg.act_dtype}  block={MINIMAX_M2_BLOCK_SIZE}")
    log0(ctx, f"  out_tol: atol={out_atol} rtol={out_rtol}")
    log0(ctx, f"  sum_tol: atol={SUM_SQ_ATOL} rtol={SUM_SQ_RTOL}")
    log0(ctx, f"  hidden={cfg.hidden_size}  local_q={cfg.num_heads_q}  local_kv={cfg.num_heads_k}")
    if not use_fp8:
        log0(ctx, f"  scheme B available: {has_staged}")
    elif cfg.tp_world > 1:
        log0(ctx, "  production FP8 TP path: W8A8 GEMM + standalone sum + comm + finalize")
    log0(ctx, "")

    failures = 0
    for num_tokens in TOKEN_COUNTS:
        if ctx.rank == 0:
            print(f"num_tokens={num_tokens}", flush=True)

        (
            hidden,
            bundle,
            positions,
            q_norm_weight,
            k_norm_weight,
            rope,
            q_norm_module,
            k_norm_module,
        ) = _make_tensors(num_tokens, cfg, ctx, seed=42 + num_tokens, fp8_linear=fp8_linear)

        ref_out, ref_sum_sq = run_eager(
            hidden, positions, bundle, q_norm_weight, k_norm_weight, rope, cfg,
            q_norm_module, k_norm_module,
        )

        out_fused, local_sum = run_fused_from_hidden(
            hidden, positions, bundle, q_norm_weight, k_norm_weight,
            rope.cos_sin_cache, cfg,
        )

        ok = True
        ok &= _check_close("fused/eager", out_fused, ref_out, out_atol, out_rtol, ctx)

        if cfg.tp_world > 1:
            assert local_sum is not None
            ok &= _check_close(
                "fused/sum_sq", local_sum, ref_sum_sq, SUM_SQ_ATOL, SUM_SQ_RTOL, ctx
            )

            if has_staged:
                out_c, local_c = run_fused_from_hidden(
                    hidden, positions, bundle, q_norm_weight, k_norm_weight,
                    rope.cos_sin_cache, cfg, use_standalone_sum_epilogue=False,
                )
                out_b, local_b = run_fused_from_hidden(
                    hidden, positions, bundle, q_norm_weight, k_norm_weight,
                    rope.cos_sin_cache, cfg, use_standalone_sum_epilogue=True,
                )
                ok &= _check_close("schemeC/out", out_c, ref_out, out_atol, out_rtol, ctx)
                ok &= _check_close(
                    "schemeC/sum", local_c, ref_sum_sq, SUM_SQ_ATOL, SUM_SQ_RTOL, ctx
                )
                ok &= _check_close("schemeB/out", out_b, ref_out, out_atol, out_rtol, ctx)
                ok &= _check_close(
                    "schemeB/sum", local_b, ref_sum_sq, SUM_SQ_ATOL, SUM_SQ_RTOL, ctx
                )
                ok &= _check_close("C_vs_B", out_c, out_b, ATOL_C_VS_B, RTOL_C_VS_B, ctx)

        fail_flag = 0 if ok else 1
        if ctx.is_distributed:
            fail_t = torch.tensor([fail_flag], device=ctx.device, dtype=torch.int32)
            dist.all_reduce(fail_t, op=dist.ReduceOp.MAX)
            fail_flag = int(fail_t.item())
            dist.barrier()
        if fail_flag:
            failures += 1

    log0(ctx, "")
    if failures:
        log0(ctx, f"FAILED: {failures} token-size case(s)")
        raise SystemExit(1)
    log0(ctx, "ALL PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fp8", "bf16"),
        default="fp8",
        help="fp8: production block W8A8 (default); bf16: incl. scheme C vs B",
    )
    args = parser.parse_args()

    if not current_platform.is_cuda_alike():
        raise SystemExit("CUDA required")

    with set_current_vllm_config(VllmConfig()):
        ctx = init_dist()
        with torch.inference_mode():
            run_accuracy(ctx, args.mode)
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
