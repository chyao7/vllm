# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone MiniMax-M2 fused QKV: qkv_proj + forward_qk + rotary_emb.

Three-stage pipeline:
  1. compute  — torch.ops._C.minimax_m2_fused_qkv_compute
  2. communicate — tensor_model_parallel_all_reduce on qk_sum_sq (TP only)
  3. finalize — torch.ops._C.minimax_m2_fused_qkv_finalize (TP only)

tp_world == 1: stage 1 fuses GEMM + sum_sq + norm + RoPE; stages 2–3 are skipped.
"""

from __future__ import annotations

import os

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.platforms import current_platform


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


def minimax_m2_fused_qkv_allocate_workspace(
    num_tokens: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Workspace for stage 2: per-token [sum(q^2), sum(k^2)] in fp32."""
    return torch.zeros(num_tokens, 2, dtype=torch.float32, device=device)


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
) -> torch.Tensor:
    """Stage 1: qkv_proj + post (fused norm/RoPE if tp_world == 1, else local sum_sq)."""
    torch.ops._C.minimax_m2_fused_qkv_compute(
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
