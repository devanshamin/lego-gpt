import gc
import warnings
from abc import abstractmethod, ABC
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from lego_gpt.blocks.language_modeling import LanguageModelingOutput, LanguageModeling


class Sampler(ABC):
    def __init__(self, model: LanguageModeling, tokenizer: PreTrainedTokenizerBase) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def encode(self, text: str):
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def decode(self, ids):
        return self.tokenizer.decode(ids.squeeze().tolist())

    @torch.inference_mode()
    def forward(self, input_ids):
        output: LanguageModelingOutput = self.model(input_ids=input_ids)
        return output.logits[:, -1, :]

    @abstractmethod 
    def generate(self):
        pass

    def clear_memory() -> None:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class GreedySampler(Sampler):

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None,
        stream: bool = False
    ) -> str:
        
        generated = ""
        input_ids = self.encode(prompt)["input_ids"]
        
        input_tokens = input_ids.shape[-1]
        model_seq_len = self.model.model.config.n_ctx
        if input_tokens + max_new_tokens > model_seq_len:
            warnings.warn(
                f"The total number of tokens ({input_tokens + max_new_tokens}) exceeds "
                f"the model's sequence length ({model_seq_len}). This may lead to truncation or "
                "unexpected behavior."
            )
            max_new_tokens = max(0, model_seq_len - input_tokens)
            print(f"Adjusted max_new_tokens to {max_new_tokens}")
        
        if stream:
            print(prompt, end="")
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_idx), dim=1)
            next_token = self.decode(next_idx)
            if stream:
                print(next_token, end="")
            generated += next_token
        
        __class__.clear_memory()

        return generated