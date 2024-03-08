import gc
from itertools import chain
from abc import abstractmethod, ABC
from typing import Optional, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from lego_gpt.blocks.language_modeling import LanguageModeling


torch.manual_seed(1234)

class SamplingConfig(BaseModel):
    temperature: float = Field(0.7, description="Temperature parameter for randomness in generation.")
    top_k: Optional[int] = Field(None, description="Top-k parameter for token sampling.")


class Sampler(ABC):

    def __init__(
        self, 
        model: LanguageModeling, 
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = next(model.parameters()).device
        self.model_size = sum([p.numel() * p.dtype.itemsize for p in chain(self.model.parameters(), self.model.buffers())])

        if max_seq_length is None:
            config = self.model.model.config
            if isinstance(config, dict):
                max_seq_length = config.get("n_ctx")
            elif isinstance(config, (tuple, BaseModel)) and hasattr(config, "n_ctx"):
                max_seq_length = config.n_ctx
            assert max_seq_length is not None, \
                "`max_seq_length` cannot be inferred! Please provide `max_seq_length` value."
        self.max_seq_length = max_seq_length

    def encode(self, text: str) -> Dict[str, Tensor]:

        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def decode(self, token_ids: Tensor, **kwargs) -> str:

        token_ids = token_ids.squeeze().tolist()
        skip_special_tokens = kwargs.get("skip_special_tokens", True)
        out = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return out

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:

        logits = self.model(input_ids=input_ids, **kwargs)[0]
        return logits

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
        logits: Tensor, # Shape: bsz, seq_len, vocab_size
        sampling_config: Optional[SamplingConfig] = None
    ) -> Tuple[Tensor, Tensor]:

        probs = Sampler.logits_to_probs(logits[0, -1], sampling_config)
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
