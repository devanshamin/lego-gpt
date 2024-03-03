import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .config import GptModelConfig
from .normalization import RMSNorm
from .attention import MultiHeadedAttention
from .feedforward import FeedForward
from .positional_embedding import precompute_freqs_cis
from lego_gpt.blocks.cache_utils import KVCache, DynamicKVCache


class DecoderBlock(nn.Module):

    def __init__(self, config: GptModelConfig, layer_idx: int):
        
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadedAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.feed_forward = FeedForward(config.n_embd, config.intermediate_size)

    def forward(
        self, 
        *,
        hidden_states: Tensor, 
        freqs_cis: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[KVCache] = None,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        
        residual = hidden_states
        hidden_states, present_key_value = self.attention(
            hidden_states=self.attention_norm(hidden_states), 
            freqs_cis=freqs_cis, 
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states)) # bsz x seq_len x n_embed
        return hidden_states, present_key_value 
    

class GptModel(nn.Module):

    def __init__(self, config: GptModelConfig):
        
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([DecoderBlock(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.freqs_cis: Optional[Tensor] = None

        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        *,
        input_ids: torch.LongTensor, 
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[Tensor, Optional[KVCache]]:

        bsz, seq_len = input_ids.shape
        use_cache = use_cache or self.config.use_cache

        if use_cache:
            if not isinstance(past_key_values, DynamicKVCache):
                past_key_values = DynamicKVCache()
            past_key_values_length = past_key_values.get_seq_length(layer_idx=0)
        else:
            past_key_values_length = 0

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, 
                seq_len + past_key_values_length, 
                dtype=torch.long, 
                device=input_ids.device
            )
        else:
            position_ids = position_ids.long()
        
        inputs_embeds = self.tok_embeddings(input_ids)
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                input_shape=(bsz, seq_len),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis(
                seq_len=self.config.n_ctx, 
                n_elem=self.config.n_embd // self.config.n_head
            )
        freqs_cis = self.freqs_cis[position_ids]
        
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states, next_kv_cache = layer(
                hidden_states=hidden_states, 
                attention_mask=attention_mask, 
                freqs_cis=freqs_cis,
                past_key_value=past_key_values, # This is a kv cache object, hence same object can be used to update the kv cache
            )

        return hidden_states, next_kv_cache