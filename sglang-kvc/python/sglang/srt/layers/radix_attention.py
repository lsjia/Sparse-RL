# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Radix attention."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing import Tuple
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils import direct_register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"

def need_compress(l: int, req_indice: int, forward_batch: ForwardBatch) -> bool:
    #print("forward_batch.compress_max_prompt: ", type(forward_batch.compress_max_prompt))
    if forward_batch.compress_algorithm is None:
        return False
    elif l <= int(forward_batch.compress_max_prompt):
        return False
    elif forward_batch.compress_algorithm in ["RKV", "snapkv", "streamingllm", "h2o", "pyramidkv"]:
        if forward_batch.forward_mode.is_extend():
            return True
        elif forward_batch.req_to_token_pool.think_forbid[req_indice]:
            return False
        elif forward_batch.forward_mode.is_decode():
            if forward_batch.compress_divide_method == "newline":
                return forward_batch.req_to_token_pool.newline_compress[req_indice]
            elif forward_batch.compress_divide_method == "step_length":
                if forward_batch.compress_divide_length is None or forward_batch.compress_divide_length == 0:
                    raise ValueError(f"Invalid compress_divide_length: {forward_batch.compress_divide_length}")
                else:
                    return forward_batch.steps % forward_batch.compress_divide_length == 0
        else:
            return False
    else:
        raise ValueError(f"Invalid compress algorithm: {forward_batch.compress_algorithm}")

class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        # ==============================================================
        # [compress] insert
        # ==============================================================
        
        # 标记变量：记录是否已经手动写入了 KV Cache
        has_saved_kv = False
        # 仅在非编译模式下尝试运行 RKV (避免打破 torch.compile)
        # 如果 get_forward_context() 不为 None，说明正在进行图捕获，跳过 Python 侧的 RKV
        is_compiling = (forward_batch.forward_mode.is_extend() and get_forward_context() is not None)
        
        if not is_compiling and getattr(forward_batch, "compress_algorithm", None) is not None:
            # 1. 缓存 Query
            self._cache_q(q, forward_batch)
            
            if forward_batch.forward_mode.is_extend():
                # Extend 阶段直接压缩 (此时 k,v 是输入参数)
                self.compress(k, v, forward_batch)
                
            elif forward_batch.forward_mode.is_decode():
                # Decode 阶段需要先写入 KV，供压缩算法读取历史
                if save_kv_cache:
                    cache_loc = (
                        forward_batch.out_cache_loc
                        if not self.is_cross_attention
                        else forward_batch.encoder_out_cache_loc
                    )
                    forward_batch.token_to_kv_pool.set_kv_buffer(self, cache_loc, k, v)
                    has_saved_kv = True # 标记已写入
                
                self.compress(k, v, forward_batch)

        # 确定传递给后端算子的 save_kv_cache 标志
        backend_save_kv = save_kv_cache and (not has_saved_kv)

        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            torch.ops.sglang.unified_attention_with_output(
                q, k, v, output, backend_save_kv, self.layer_id, **kwargs
            )
            return output
        else:
            return forward_batch.attn_backend.forward(
                q,
                k,
                v,
                self,
                forward_batch,
                backend_save_kv,
                **kwargs,
            )

    def _get_metadata(self, idx: int, forward_batch: ForwardBatch) -> Tuple[int, int, int]:
        """
        Get forward_batch's necessary metadata for compression algorithms
        """
        if forward_batch.forward_mode.is_extend():
            req_indice = forward_batch.req_pool_indices[idx].item()
            seq_len = int(forward_batch.seq_lens[idx])
            prefix_len = forward_batch.extend_prefix_lens[idx].item()
            cur_len = seq_len - prefix_len
            return req_indice, seq_len, cur_len
        elif forward_batch.forward_mode.is_decode():
            req_indice = forward_batch.req_pool_indices[idx].item()
            seq_len = int(forward_batch.seq_lens[idx])
            prefix_len = 0
            cur_len = seq_len - prefix_len
            return req_indice, seq_len, cur_len

    def _reshape_qkv_compress(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape q, k, v tensors to match KV compression cluster's input. 

        Input shape: (seq_len, num_heads, head_dim)

        Output shape: (batch_size=1, num_heads, seq_len, head_dim)
        """
        q_state = q.view(-1, self.tp_q_head_num, self.head_dim).transpose(0, 1).unsqueeze(0)[..., -forward_batch.compress_max_window : , :]
        k_state = k.transpose(0, 1).unsqueeze(0)
        v_state = v.transpose(0, 1).unsqueeze(0)
        return q_state, k_state, v_state
    
    def _cache_q(self, q: torch.Tensor, forward_batch: ForwardBatch) -> None:
        """
        cache q tensor for compression use
        """
        if forward_batch.forward_mode.is_extend():
            # For extend/prefill, Q contains all seq_len, need to seperate it one by one
            layer_id = self.layer_id
            start = end = 0
            for i in range(len(forward_batch.req_pool_indices)):
                req_indice, _, cur_len = self._get_metadata(i, forward_batch)
                end = start + cur_len - 1
                # we only need to retain the last 32 q tensor for current request
                forward_batch.req_to_token_pool.q_cache[layer_id][req_indice] = q[start : end + 1][-int(forward_batch.req_to_token_pool.max_window) : , ...]
                start = end + 1
            assert sum(forward_batch.seq_lens) - sum(forward_batch.extend_prefix_lens) == q.shape[0]
        elif forward_batch.forward_mode.is_decode():
            layer_id = self.layer_id
            # For decode, every indice has only one q tensor
            for i in range(len(forward_batch.req_pool_indices)):
                req_indice = forward_batch.req_pool_indices[i].item()
                old_tensor = forward_batch.req_to_token_pool.q_cache[layer_id][req_indice]
                forward_batch.req_to_token_pool.q_cache[layer_id][req_indice] = torch.cat((old_tensor, q[i].unsqueeze(0)), dim=0)[-int(forward_batch.req_to_token_pool.max_window) : , ...]
                del old_tensor
            assert q.shape[0] == len(forward_batch.req_pool_indices)

    def compress(self, k: torch.Tensor, v: torch.Tensor, forward_batch: ForwardBatch) -> None:
        """
        Modified Compress: 
        1. Prefill (Extend): ALWAYS keep full KV (Initialize).
        2. Decode: Run compression or append new token.
        """
        # initialize
        if getattr(forward_batch, "retained_token_positions", None) is None:
            forward_batch.retained_token_positions = {}

        if not hasattr(forward_batch.req_to_token_pool, "retained_indices_cache"):
            forward_batch.req_to_token_pool.retained_indices_cache = {}
        
        pool_cache = forward_batch.req_to_token_pool.retained_indices_cache

        # === Extend / Prefill Phase (uncompressed) ===
        if forward_batch.forward_mode.is_extend():
            # Traverse all req in batch
            for i in range(len(forward_batch.req_pool_indices)):
                req_indice, seq_len, _ = self._get_metadata(i, forward_batch)
                
                # get kv loc indices
                kv_locs = forward_batch.req_to_token_pool.req_to_token[req_indice][:seq_len]
                
                forward_batch.retained_token_positions[req_indice] = kv_locs
                
                # B. initialize cache
                pool_cache[req_indice] = kv_locs
                

        # === Decode / Generation Phase ===
        elif forward_batch.forward_mode.is_decode():
            for i in range(len(forward_batch.req_pool_indices)):
                req_indice, seq_len, _ = self._get_metadata(i, forward_batch)
                
                # get the new token at the current step (seq_len includes the current step)
                curr_token_physical_idx = forward_batch.req_to_token_pool.req_to_token[req_indice][seq_len - 1].unsqueeze(0)
                
                # Check if compression is triggered
                is_triggered = need_compress(seq_len, req_indice, forward_batch)

                # --- prepare candidate pool ---
                if req_indice in pool_cache:
                    prev_kept_indices = pool_cache[req_indice]
                    # Simple deduplication check: Prevents duplicate additions of the same token
                    if len(prev_kept_indices) > 0 and prev_kept_indices[-1] == curr_token_physical_idx:
                        candidate_physical_indices = prev_kept_indices
                    else:
                        candidate_physical_indices = torch.cat([prev_kept_indices, curr_token_physical_idx])
                else:
                    candidate_physical_indices = forward_batch.req_to_token_pool.req_to_token[req_indice][:seq_len]

                if is_triggered:
                    # Case A: compression is triggered
                    
                    # 1. read KV (Budget part)
                    k_state = forward_batch.token_to_kv_pool.k_buffer[self.layer_id][candidate_physical_indices]
                    v_state = forward_batch.token_to_kv_pool.v_buffer[self.layer_id][candidate_physical_indices]
                    
                    # 2. read Q
                    q_state = forward_batch.req_to_token_pool.q_cache[self.layer_id][req_indice]
                    
                    # 3. calculation
                    q_state, k_state, v_state = self._reshape_qkv_compress(q_state, k_state, v_state, forward_batch)
                    relative_indices = forward_batch.compress_cluster.get_compress_indices(k_state, q_state, v_state, None)
                    
                    # 4. mapping
                    new_retained = candidate_physical_indices[relative_indices]
                    
                    # 5. update
                    pool_cache[req_indice] = new_retained
                    forward_batch.retained_token_positions[req_indice] = new_retained
                    
                else:
                    # Case B: not triggered
                    pool_cache[req_indice] = candidate_physical_indices
                    forward_batch.retained_token_positions[req_indice] = candidate_physical_indices
                                      
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks

    ret = forward_batch.attn_backend.forward(
        query, key, value, attention_layer, forward_batch, save_kv_cache, **kwargs
    )
    assert (
        output.numel() == ret.numel()
    ), f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"

    output.view(ret.shape).copy_(ret)
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
)
