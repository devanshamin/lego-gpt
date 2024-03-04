import warnings
from typing import Optional

import torch

from lego_gpt.inference.base_sampler import Sampler, SamplingConfig


class GreedySampler(Sampler):

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int, 
        sampling_config: Optional[SamplingConfig] = None,
        stream: bool = False
    ) -> str:
        
        cls = __class__
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
            next_idx, _ = cls.sample(logits, sampling_config)
            input_ids = torch.cat((input_ids, next_idx), dim=1)
            next_token = self.decode(next_idx)
            if stream:
                print(next_token, end="")
            generated += next_token
        
        cls.clear_memory()

        return generated