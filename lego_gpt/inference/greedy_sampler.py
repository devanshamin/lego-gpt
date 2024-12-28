import time
import warnings
from typing import Optional, List, Tuple

import torch
from torch import Tensor

from lego_gpt.inference.base_sampler import Sampler, SamplingConfig


class GreedySampler(Sampler):
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        sampling_config: Optional[SamplingConfig] = None,
        use_cache: bool = True
    ) -> str:

        input_ids = self.encode(prompt)["input_ids"]
        prompt_length = input_ids.shape[-1]
        total_length = prompt_length + max_new_tokens

        if total_length > self.max_seq_length:
            warnings.warn(
                f"The total number of tokens ({total_length}) exceeds "
                f"the model's sequence length ({self.max_seq_length}). This may lead to truncation or "
                "unexpected behavior."
            )
            max_new_tokens = max(0, self.max_seq_length - prompt_length)
            print(f"Adjusted max_new_tokens to {max_new_tokens}")

        start_time = time.perf_counter()
        new_token_ids, _ = self.generate_n_tokens(
            input_ids=input_ids,
            num_new_tokens=max_new_tokens,
            sampling_config=sampling_config,
            use_cache=use_cache
        )
        time_taken = time.perf_counter() - start_time
        if self.verbose:
            tokens_per_sec = max_new_tokens / time_taken
            print(
                f"Time for inference: {time_taken:.02f} sec total, {tokens_per_sec:.02f} tokens/sec",
                f"Bandwidth achieved: {self.model_size * tokens_per_sec / 1e9:.02f} GB/s",
                sep="\n"
            )
        new_token_ids = torch.cat(new_token_ids)
        output = self.decode(token_ids=torch.cat((input_ids.squeeze(), new_token_ids)))
        GreedySampler.clear_memory()

        return output

    def generate_n_tokens(
        self,
        input_ids: Tensor,
        num_new_tokens: int,
        **kwargs
    ) -> Tuple[List[Tensor], List[Tensor]]:

        new_token_ids, new_probs = [], []
        for _ in range(num_new_tokens):
            next_token_id, probs = self.generate_one_token(input_ids.view(1, -1), **kwargs)
            new_token_ids.append(next_token_id.clone())
            new_probs.append(probs.clone())
            input_ids = next_token_id.view(1, -1)
        return new_token_ids, new_probs
