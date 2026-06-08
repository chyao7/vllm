# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)


class Qwen3Attention(nn.Module):
    _logged_qk_norm_rope_execution: str | None = None

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.use_custom_qkv_proj = self._can_use_custom_qkv_proj(
            rope_parameters, dual_chunk_attention_config
        )
        self.use_fused_kv_cache_write = self.use_custom_qkv_proj

    def _can_use_custom_qkv_proj(
        self,
        rope_parameters: dict | None,
        dual_chunk_attention_config: dict[str, Any] | None,
    ) -> bool:
        import vllm._C  # noqa: F401 — register torch.ops._C

        if not current_platform.is_cuda_alike():
            return False
        if not hasattr(torch.ops._C, "custom_qkv_proj_rmsnorm_rope"):
            return False
        if not hasattr(torch.ops._C, "custom_qkj_proj_rmsnormal_reshap_cache"):
            return False
        if not isinstance(self.qkv_proj.quant_method, UnquantizedLinearMethod):
            return False
        if dual_chunk_attention_config is not None:
            return False
        rope_parameters = rope_parameters or {}
        if rope_parameters.get("rope_type", "default") != "default":
            return False
        if rope_parameters.get("partial_rotary_factor", 1.0) != 1.0:
            return False
        if "mrope_section" in rope_parameters:
            return False
        if not getattr(self.rotary_emb, "is_neox_style", False):
            return False
        return True

    @staticmethod
    def _split_flash_kv_cache(
        kv_cache: torch.Tensor, layer_name: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Split stacked KV cache into key/value views for Flash NHD layout.

        Per-layer cache is usually [2, num_blocks, block_size, num_kv_heads, H].
        Cross-layer uniform buffers may be [num_layers, 2, num_blocks, ...].
        """
        if kv_cache.numel() == 0 or kv_cache.dim() < 4:
            return None, None
        if kv_cache.shape[0] == 2:
            return kv_cache.unbind(0)
        if kv_cache.shape[1] == 2:
            return kv_cache.unbind(1)
        if kv_cache.dim() >= 5 and kv_cache.shape[1] == 2:
            layer_idx = extract_layer_index(layer_name)
            if layer_idx < 0 or layer_idx >= kv_cache.shape[0]:
                return None, None
            return kv_cache[layer_idx].unbind(0)
        return None, None

    def _get_fused_kv_cache_write_tensors(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.use_fused_kv_cache_write:
            return None, None, None
        try:
            _, _, kv_cache, layer_slot_mapping = get_attention_context(
                self.attn.layer_name
            )
        except (AssertionError, AttributeError, KeyError):
            return None, None, None
        if layer_slot_mapping is None:
            return None, None, None
        if layer_slot_mapping.size(0) < num_tokens:
            return None, None, None
        key_cache, value_cache = self._split_flash_kv_cache(
            kv_cache, self.attn.layer_name
        )
        if key_cache is None or value_cache is None:
            return None, None, None
        slot_mapping = layer_slot_mapping[:num_tokens]
        return key_cache, value_cache, slot_mapping

    def _get_qkv_workspace(
        self, num_tokens: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        out_dim = self.q_size + 2 * self.kv_size
        buf = getattr(self, "_qkv_workspace", None)
        if (
            buf is None
            or buf.size(0) < num_tokens
            or buf.dtype != dtype
            or buf.device != device
        ):
            self._qkv_workspace = torch.empty(
                num_tokens, out_dim, dtype=dtype, device=device
            )
        return self._qkv_workspace[:num_tokens]

    def _custom_qkv_proj_rmsnorm_rope(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.size(0)
        qkv = self._get_qkv_workspace(
            num_tokens, hidden_states.dtype, hidden_states.device
        )
        qkv_bias = (
            None
            if self.qkv_proj.bias is None or self.qkv_proj.skip_bias_add
            else self.qkv_proj.bias
        )
        key_cache, value_cache, slot_mapping = self._get_fused_kv_cache_write_tensors(
            num_tokens
        )
        fused_cache = key_cache is not None
        log_tag = (
            "custom_qkj_proj_rmsnormal_reshap_cache"
            if fused_cache
            else "custom_qkv_proj_rmsnorm_rope"
        )
        cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(hidden_states)
        is_neox = self.rotary_emb.is_neox_style
        if Qwen3Attention._logged_qk_norm_rope_execution != log_tag:
            logger.info(
                "Qwen3Attention: executing %s "
                "(fused_kv_cache_write=%s, num_tokens=%s, hidden=%s, "
                "num_heads_q=%s, num_heads_k=%s, head_dim=%s, rotary_dim=%s, "
                "is_neox=%s)",
                log_tag,
                fused_cache,
                num_tokens,
                hidden_states.size(-1),
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
                is_neox,
            )
            Qwen3Attention._logged_qk_norm_rope_execution = log_tag
        if fused_cache:
            assert key_cache is not None
            assert value_cache is not None
            assert slot_mapping is not None
            ops.custom_qkj_proj_rmsnormal_reshap_cache(
                qkv,
                hidden_states,
                self.qkv_proj.weight,
                qkv_bias,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
                self.q_norm.variance_epsilon,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                is_neox,
                positions.view(-1),
                key_cache,
                value_cache,
                slot_mapping,
            )
        else:
            ops.custom_qkv_proj_rmsnorm_rope(
                qkv,
                hidden_states,
                self.qkv_proj.weight,
                qkv_bias,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
                self.q_norm.variance_epsilon,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                is_neox,
                positions.view(-1),
            )
        q = qkv[:, : self.q_size]
        if fused_cache:
            # K/V written in custom op; pass None so Attention skips
            # unified_kv_cache_update (reshape_and_cache_flash).
            return q, None, None
        # Fallback when slot_mapping/kv_cache unavailable (e.g. profiling).
        k = qkv[:, self.q_size : self.q_size + self.kv_size]
        v = qkv[:, self.q_size + self.kv_size :]
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_custom_qkv_proj:
            q, k, v = self._custom_qkv_proj_rmsnorm_rope(hidden_states, positions)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Add qk-norm
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


class Qwen3ForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
