import gc
from abc import abstractmethod, ABC
from typing import Optional, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from lego_gpt.blocks.language_modeling import LanguageModelingOutput, LanguageModeling


class SamplingConfig(BaseModel):
    temperature: float = 1.0
    top_k: Optional[int] = None


class Sampler(ABC):

    def __init__(self, model: LanguageModeling, tokenizer: PreTrainedTokenizerBase) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def encode(self, text: str) -> Dict[str, Tensor]:

        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def decode(self, ids: Tensor) -> str:

        return self.tokenizer.decode(ids.squeeze().tolist())

    @torch.inference_mode()
    def forward(self, input_ids) -> Tensor:

        output: LanguageModelingOutput = self.model(input_ids=input_ids)
        return output.logits[:, -1, :]

    @staticmethod
    def logits_to_probs(
        logits: Tensor, 
        sampling_config: Optional[SamplingConfig] = None
    ) -> Tensor:
        
        cfg = sampling_config or SamplingConfig()
        logits = logits / max(cfg.temperature, 1e-5)
        if cfg.top_k is not None:
            v, _ = torch.topk(logits, k=min(cfg.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)   
        return probs

    @staticmethod
    def multinomial_sample_one_no_sync(probs: Tensor) -> Tensor:
        """Optimized multinomial sampling without a GPU<->CPU sync."""

        q = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

    @staticmethod
    def sample(
        logits: Tensor, 
        sampling_config: Optional[SamplingConfig] = None
    ) -> Tuple[Tensor, Tensor]:
        
        probs = Sampler.logits_to_probs(logits, sampling_config)
        next_idx = Sampler.multinomial_sample_one_no_sync(probs)
        return next_idx, probs

    @abstractmethod 
    def generate(self):
        pass

    @staticmethod
    def clear_memory() -> None:
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
