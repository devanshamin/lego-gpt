import gc
from abc import abstractmethod, ABC
from typing import Optional, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from lego_gpt.blocks.language_modeling import LanguageModelingOutput, LanguageModeling


torch.manual_seed(1234)

class SamplingConfig(BaseModel):
    temperature: float = Field(0.7, description="Temperature parameter for randomness in generation.")
    top_k: Optional[int] = Field(None, description="Top-k parameter for token sampling.")


class Sampler(ABC):

    def __init__(self, model: LanguageModeling, tokenizer: PreTrainedTokenizerBase) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_seq_length = self.model.model.config.n_ctx

    def encode(self, text: str) -> Dict[str, Tensor]:

        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def decode(self, token_ids: Tensor, **kwargs) -> str:

        token_ids = token_ids.squeeze().tolist()
        skip_special_tokens = kwargs.get("skip_special_tokens", True)
        out = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return out

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:

        output: LanguageModelingOutput = self.model(input_ids=input_ids, **kwargs)
        return output.logits[0, -1]

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
        next_token_id = Sampler.multinomial_sample_one_no_sync(probs)
        return next_token_id, probs

    def generate_one_token(
        self, 
        input_ids: Tensor, 
        sampling_config: Optional[SamplingConfig] = None, 
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        logits = self.forward(input_ids, **kwargs)
        return Sampler.sample(logits, sampling_config)

    @abstractmethod 
    def generate(self):
        pass

    @staticmethod
    def clear_memory() -> None:
        
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
