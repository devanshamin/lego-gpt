import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .config import GptModelConfig
from .positional_embedding import apply_rotary_emb
from lego_gpt.blocks.cache_utils import KVCache


class MultiHeadedAttention(nn.Module):

    def __init__(self, config: GptModelConfig, layer_idx: int):
        
        super().__init__()
        self.layer_idx = layer_idx
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_local_heads = config.n_local_heads
        self.n_embd = config.n_embd

        # Q -> config.n_head
        # K, V -> self.n_local_heads
        total_head_dim = (config.n_head + 2 * self.n_local_heads) * self.head_dim
        self.wqkv = nn.Linear(config.n_embd, total_head_dim, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.is_flash_available = hasattr(F, "scaled_dot_product_attention")
        if not self.is_flash_available:
            print("WARNING: Using slow attention! Flash Attention requires PyTorch >= 2.0")
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.attn_pdrop = config.attn_pdrop

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):

        if prefix + "wq.weight" in state_dict:
            # Match the named parameters in the init constructor
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self, 
        *,
        hidden_states: Tensor, 
        freqs_cis: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[KVCache] = None,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        
        bsz, seq_len, n_embd = hidden_states.size()

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(hidden_states).split([self.n_embd, kv_size, kv_size], dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim)
        k = k.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(layer_idx=self.layer_idx)
        
        if past_key_value is not None:
            k, v = past_key_value.update(key_states=k, value_states=v, layer_idx=self.layer_idx)

        # Repeat slices of the tensor along the local head (dim=1) dimension
        # This is done to match the shape of the query (q)
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        if self.is_flash_available:
            attn_output = F.scaled_dot_product_attention(
                query=q, 
                key=k, 
                value=v, 
                attn_mask=attention_mask, 
                dropout_p=self.attn_pdrop if self.training else 0.0,
                is_causal=True
            )
        else:
            attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # bsz x n_head x seq_len x seq_len
            if attention_mask is not None:
                # Attention mask -> orig. attention mask (0 & 1) + causal tril
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = attn_weights @ v # bsz x n_head x seq_len x head_dim

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, n_embd)
        attn_output = self.resid_dropout(self.wo(attn_output))
        return attn_output, past_key_value