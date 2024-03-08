from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lego_gpt import blocks
from lego_gpt.blocks.cache_utils import KVCache
from lego_gpt.blocks.transformer.normalization import RMSNorm
from lego_gpt.blocks.base import BaseLanguageModelConfig, BaseLanguageModel


class LanguageModeling(BaseLanguageModel):

    def __init__(self, config: BaseLanguageModelConfig):

        super().__init__()
        model_cls = getattr(blocks, config.architecture, None)
        assert model_cls is not None, f"Model class with name '{config.architecture}' not found!"
        self.model = model_cls(config)
        self.norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        labels: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:
        
        hidden_states, next_kv_cache = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states) # bsz x seq_len x vocab_size
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # bsz * seq_len x vocab_size
                labels.view(-1), # bsz * seq_len
            )
        else:
            loss = None

        # Pydantic class is not used here for wrapping `LanguageModeling`'s output,
        # since it causes issues with the JIT compiler used by `torch.compile` function        
        return logits, loss, next_kv_cache
