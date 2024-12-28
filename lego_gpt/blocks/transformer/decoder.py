from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from .config import GptModelConfig
from .normalization import RMSNorm
from .attention import MultiHeadedAttention
from .feedforward import FeedForward
from lego_gpt.blocks.cache_utils import KVCache


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
