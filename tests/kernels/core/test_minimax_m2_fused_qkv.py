# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for standalone MiniMax-M2 minimax_m2_fused_qkv_from_hidden."""

import sys
from pathlib import Path

_VLLM_ROOT = Path(__file__).resolve().parents[3]
if str(_VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLLM_ROOT))

import pytest
import torch

import vllm._C  # noqa: F401

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.minimax_m2_fused_qkv import (
    minimax_m2_fused_qkv_available,
    minimax_m2_fused_qkv_from_hidden,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _reference_qkv_from_hidden(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    eps: float,
    tp_world: int = 1,
) -> torch.Tensor:
    """Match MiniMaxText01RMSNormTP.forward_qk + rotary_emb (tp_world=1)."""
    qkv = torch.nn.functional.linear(hidden, weight, bias)
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    orig_dtype = q.dtype
    q = q.float()
    k = k.float()
    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)
    if tp_world > 1:
        qk_var = torch.cat([q_var, k_var], dim=-1)
        qk_var = qk_var / tp_world
        q_var, k_var = qk_var.chunk(2, dim=-1)
    q = q * torch.rsqrt(q_var + eps) * q_norm_weight.float()
    k = k * torch.rsqrt(k_var + eps) * k_norm_weight.float()
    q = q.to(orig_dtype)
    k = k.to(orig_dtype)
    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


def _run_single_gpu_test() -> None:
    if not current_platform.is_cuda_alike():
        return
    if not minimax_m2_fused_qkv_available():
        return

    device = "cuda:0"
    dtype = torch.bfloat16
    eps = 1e-6
    set_random_seed(7)

    num_heads_q, num_heads_kv, head_dim = 8, 2, 128
    hidden_size = 512
    num_tokens = 4
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim
    out_dim = q_size + 2 * kv_size

    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(out_dim, hidden_size, dtype=dtype, device=device)
    qkv = torch.empty(num_tokens, out_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm_weight = torch.ones(q_size, device=device, dtype=dtype)
    k_norm_weight = torch.ones(kv_size, device=device, dtype=dtype)
    q_norm_weight.normal_(mean=1.0, std=0.1)
    k_norm_weight.normal_(mean=1.0, std=0.1)

    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=True,
        dtype=dtype,
    ).to(device)

    minimax_m2_fused_qkv_from_hidden(
        hidden_states=hidden,
        positions=positions,
        qkv=qkv,
        qkv_weight=weight,
        qkv_bias=None,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=rope.cos_sin_cache,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_kv,
        num_heads_v=num_heads_kv,
        head_dim=head_dim,
        eps=eps,
        is_neox=True,
        tp_world=1,
    )

    ref = _reference_qkv_from_hidden(
        hidden,
        positions,
        weight,
        None,
        q_norm_weight,
        k_norm_weight,
        rope,
        num_heads_q,
        num_heads_kv,
        head_dim,
        eps,
        tp_world=1,
    )

    torch.testing.assert_close(qkv, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.usefixtures("default_vllm_config")
@torch.inference_mode()
def test_minimax_m2_fused_qkv_single_gpu():
    _run_single_gpu_test()


if __name__ == "__main__":
    if not current_platform.is_cuda_alike():
        raise SystemExit("CUDA required")
    with set_current_vllm_config(VllmConfig()):
        _run_single_gpu_test()
    print("PASS: minimax_m2_fused_qkv")
