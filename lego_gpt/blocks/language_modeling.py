from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from lego_gpt import blocks
from lego_gpt.blocks.cache_utils import KVCache
from lego_gpt.blocks.transformer.normalization import RMSNorm
from lego_gpt.blocks.base import BaseLanguageModelConfig, BaseLanguageModel


class LanguageModelingOutput(BaseModel):
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[KVCache] = None

    class Config: 
        arbitrary_types_allowed = True


class LanguageModeling(BaseLanguageModel):

    def __init__(self, config: BaseLanguageModelConfig):

        super().__init__()
        model_cls = getattr(blocks, config.architecture, None)
        assert model_cls is not None, f"Model class with name '{config.architecture}' not found!"
        self.model = model_cls(config)
        self.norm = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._register_load_state_dict_pre_hook(self.load_hook)
    
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def load_hook(self, state_dict, *args):
        
        params = list(state_dict.keys())
        for name in params:
            if name == "output.weight":
                state_dict["lm_head.weight"] = state_dict.pop(name)
            elif name.startswith("layers") or name.startswith("tok_embeddings"):
                state_dict["model." + name] = state_dict.pop(name)

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        labels: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
    ) -> LanguageModelingOutput:
        
        hidden_states, next_kv_cache = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = self.norm(hidden_states)
        if labels is None:
            logits = self.lm_head(hidden_states[:, [-1], :]) # bsz x 1 x vocab_size
            loss = None
        else:
            logits = self.lm_head(hidden_states) # bsz x seq_len x vocab_size
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # bsz * seq_len x vocab_size
                labels.view(-1), # bsz * seq_len
            )
        
        return LanguageModelingOutput(
            logits=logits,
            loss=loss,
            past_key_values=next_kv_cache
        )