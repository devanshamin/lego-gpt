from typing import List, Tuple
from abc import ABC, abstractmethod

import torch


class KVCache(ABC):

    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_seq_length(self, layer_idx: int) -> int:
        pass


class DynamicKVCache(KVCache):

    def __init__(self) -> None:
        
        # KV cache where each tensor has shape: bsz x n_head x seq_len x head_dim 
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0  # Keep track of no. of tokens seen by the cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        if layer_idx < len(self):
            # Concat along the seq_len dimension
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        else:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __getitem__(self, layer_idx: int):

        if layer_idx < len(self):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}!")
        
    def __len__(self) -> int:

        return len(self.key_cache)

    def get_seq_length(self, layer_idx: int) -> int:

        return self.key_cache[layer_idx].shape[-2] if layer_idx < len(self) else 0
