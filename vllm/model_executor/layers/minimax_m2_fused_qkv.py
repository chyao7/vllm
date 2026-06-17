# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone MiniMax-M2 fused QKV: qkv_proj + forward_qk + rotary_emb.

Three-stage pipeline (tp_world > 1):
  1. compute  — FP8 block GEMM (W8A8) or BF16 CUTLASS GEMM + epilogue
                (bias + local sum_sq -> qk_sum_sq)
  2. communicate — tensor_model_parallel_all_reduce on qk_sum_sq
  3. finalize — token kernel: read q/k once -> norm + NeoX RoPE

tp_world == 1: GEMM (+ bias) then one token kernel (load q/k once in SMEM,
               sum + norm + RoPE); stages 2–3 are skipped.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        W8A8BlockFp8LinearOp,
    )

_FP8_WEIGHT_DTYPES = frozenset(
    {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    }
)


def minimax_m2_fused_qkv_enabled() -> bool:
    return os.environ.get("VLLM_MINIMAX_M2_FUSED_QKV", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def minimax_m2_fused_qkv_available() -> bool:
    return current_platform.is_cuda_alike() and hasattr(
        torch.ops._C, "minimax_m2_fused_qkv_compute"
    ) and hasattr(torch.ops._C, "minimax_m2_fused_qkv_finalize")


def minimax_m2_fused_qkv_fp8_post_available() -> bool:
    return hasattr(torch.ops._C, "minimax_m2_fused_qkv_post_fused")


def minimax_m2_fused_qkv_staged_benchmark_available() -> bool:
    """Scheme B microbench ops (gemm-only / standalone sum / compute flag)."""
    if not minimax_m2_fused_qkv_available():
        return False
    if not hasattr(torch.ops._C, "minimax_m2_fused_qkv_gemm_only"):
        return False
    if not hasattr(torch.ops._C, "minimax_m2_fused_qkv_epilogue_sum_sq"):
        return False
    try:
        schema = str(torch.ops._C.minimax_m2_fused_qkv_compute.default._schema)
    except Exception:
        return False
    return "use_standalone_sum_epilogue" in schema


def _is_fp8_qkv_weight(qkv_weight: torch.Tensor) -> bool:
    return qkv_weight.dtype in _FP8_WEIGHT_DTYPES


def _call_fused_qkv_compute(op, *, use_standalone_sum_epilogue: bool, args: tuple) -> None:
    if use_standalone_sum_epilogue and minimax_m2_fused_qkv_staged_benchmark_available():
        op(*args, use_standalone_sum_epilogue)
    else:
        op(*args)


def minimax_m2_fused_qkv_allocate_workspace(
    num_tokens: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Workspace for stage 2: per-token [sum(q^2), sum(k^2)] in fp32."""
    return torch.zeros(num_tokens, 2, dtype=torch.float32, device=device)


def _run_fp8_qkv_gemm(
    hidden_states: torch.Tensor,
    qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    fp8_block_linear: W8A8BlockFp8LinearOp,
    qkv_bias: torch.Tensor | None,
    tp_world: int,
) -> bool:
    """Block FP8 GEMM aligned with Fp8LinearMethod. Returns defer_bias flag."""
    defer_bias = tp_world > 1 and qkv_bias is not None
    out = fp8_block_linear.apply(
        input=hidden_states,
        weight=qkv_weight,
        weight_scale=qkv_weight_scale,
        bias=None if defer_bias else qkv_bias,
    )
    qkv.copy_(out)
    return defer_bias


def _post_fused_args(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    is_neox: bool,
    qk_sum_sq: torch.Tensor,
) -> tuple:
    return (
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions.view(-1),
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        is_neox,
        qk_sum_sq,
    )


def minimax_m2_fused_qkv_compute(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    qk_sum_sq: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    is_neox: bool,
    tp_world: int,
    use_standalone_sum_epilogue: bool = False,
    qkv_weight_scale: torch.Tensor | None = None,
    fp8_block_linear: W8A8BlockFp8LinearOp | None = None,
) -> torch.Tensor:
    """Stage 1: GEMM + (tp>1: local sum_sq | tp==1: fused norm/RoPE)."""
    if _is_fp8_qkv_weight(qkv_weight):
        if qkv_weight_scale is None or fp8_block_linear is None:
            raise ValueError(
                "FP8 qkv_weight requires qkv_weight_scale and fp8_block_linear"
            )
        if not minimax_m2_fused_qkv_fp8_post_available():
            raise RuntimeError(
                "minimax_m2_fused_qkv_post_fused op not found; rebuild vLLM"
            )
        defer_bias = _run_fp8_qkv_gemm(
            hidden_states,
            qkv,
            qkv_weight,
            qkv_weight_scale,
            fp8_block_linear,
            qkv_bias,
            tp_world,
        )
        q_size = num_heads_q * head_dim
        kv_size = num_heads_k * head_dim
        if tp_world > 1:
            minimax_m2_fused_qkv_epilogue_sum_sq(
                qkv,
                qk_sum_sq,
                q_size,
                kv_size,
                qkv_bias=qkv_bias if defer_bias else None,
            )
        else:
            torch.ops._C.minimax_m2_fused_qkv_post_fused(
                *_post_fused_args(
                    qkv,
                    q_norm_weight,
                    k_norm_weight,
                    cos_sin_cache,
                    positions,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_dim,
                    eps,
                    is_neox,
                    qk_sum_sq,
                )
            )
        return qkv

    _call_fused_qkv_compute(
        torch.ops._C.minimax_m2_fused_qkv_compute,
        use_standalone_sum_epilogue=use_standalone_sum_epilogue,
        args=(
            qkv,
            hidden_states,
            qkv_weight,
            qkv_bias,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions.view(-1),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            is_neox,
            tp_world,
            qk_sum_sq,
        ),
    )
    return qkv


def minimax_m2_fused_qkv_gemm_only(
    hidden_states: torch.Tensor,
    qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    qkv_weight_scale: torch.Tensor | None = None,
    fp8_block_linear: W8A8BlockFp8LinearOp | None = None,
) -> torch.Tensor:
    """Local qkv_proj GEMM only (FP8 block / BF16 CUTLASS, no sum/norm/RoPE)."""
    if _is_fp8_qkv_weight(qkv_weight):
        if qkv_weight_scale is None or fp8_block_linear is None:
            raise ValueError(
                "FP8 qkv_weight requires qkv_weight_scale and fp8_block_linear"
            )
        _run_fp8_qkv_gemm(
            hidden_states,
            qkv,
            qkv_weight,
            qkv_weight_scale,
            fp8_block_linear,
            qkv_bias,
            tp_world=1,
        )
        return qkv

    torch.ops._C.minimax_m2_fused_qkv_gemm_only(
        qkv, hidden_states, qkv_weight, qkv_bias
    )
    return qkv


def minimax_m2_fused_qkv_epilogue_sum_sq(
    qkv: torch.Tensor,
    qk_sum_sq: torch.Tensor,
    q_size: int,
    kv_size: int,
    qkv_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standalone GEMM epilogue: bias (optional) + local sum(q^2/k^2)."""
    torch.ops._C.minimax_m2_fused_qkv_epilogue_sum_sq(
        qkv, qk_sum_sq, qkv_bias, q_size, kv_size
    )
    return qkv


def minimax_m2_fused_qkv_communicate(
    qk_sum_sq: torch.Tensor,
    tp_world: int,
) -> torch.Tensor:
    """Stage 2: all_reduce local sum-of-squares across TP ranks (no-op if tp_world <= 1)."""
    if tp_world > 1:
        tensor_model_parallel_all_reduce(qk_sum_sq)
    return qk_sum_sq


def minimax_m2_fused_qkv_finalize(
    positions: torch.Tensor,
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    qk_sum_sq: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    is_neox: bool,
    tp_world: int,
) -> torch.Tensor:
    """Stage 3: norm + RoPE from global sum_sq (required when tp_world > 1)."""
    torch.ops._C.minimax_m2_fused_qkv_finalize(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions.view(-1),
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        is_neox,
        tp_world,
        qk_sum_sq,
    )
    return qkv


def minimax_m2_fused_qkv_from_hidden(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qkv: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    is_neox: bool,
    tp_world: int | None = None,
    qkv_weight_scale: torch.Tensor | None = None,
    fp8_block_linear: W8A8BlockFp8LinearOp | None = None,
) -> torch.Tensor:
    """Run compute -> communicate -> finalize when TP requires it."""
    if tp_world is None:
        tp_world = get_tensor_model_parallel_world_size()

    qk_sum_sq = minimax_m2_fused_qkv_allocate_workspace(
        hidden_states.size(0), qkv.device
    )

    minimax_m2_fused_qkv_compute(
        hidden_states=hidden_states,
        positions=positions,
        qkv=qkv,
        qkv_weight=qkv_weight,
        qkv_bias=qkv_bias,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        qk_sum_sq=qk_sum_sq,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=eps,
        is_neox=is_neox,
        tp_world=tp_world,
        qkv_weight_scale=qkv_weight_scale,
        fp8_block_linear=fp8_block_linear,
    )

    if tp_world > 1:
        minimax_m2_fused_qkv_communicate(qk_sum_sq, tp_world)
        minimax_m2_fused_qkv_finalize(
            positions=positions,
            qkv=qkv,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            qk_sum_sq=qk_sum_sq,
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            eps=eps,
            is_neox=is_neox,
            tp_world=tp_world,
        )

    return qkv
