import json
from typing import Optional
from pathlib import Path

from lego_gpt.blocks.base import BaseLanguageModelConfig


def _find_multiple(n: int, k: int) -> int:

    if n % k == 0:
        return n
    return n + k - (n % k)


class GptModelConfig(BaseLanguageModelConfig):
    vocab_size: int
    architecture: str = "GptModel"
    n_ctx: int = 512 # GptModel
    n_embd: int = 768 # GptModel 
    n_layer: int = 12 # GptModel (DecoderBlock layers)
    n_head: int = 12 # MultiHeadedAttention
    n_local_heads: int = -1 # MultiQueryAttention (n_local_heads == 1); GroupedQueryAttention (1 < n_local_heads < n_head)
    attn_pdrop: float = 0.1 # MultiHeadedAttention
    resid_pdrop: float = 0.1 # MultiHeadedAttention & FeedForward
    layer_norm_epsilon: float = 1e-05 # DecoderBlock
    rope_theta: int = 10000 # MultiHeadedAttention
    use_cache: bool = True # GptModel; Used to speed up sequential decoding

    intermediate_size: Optional[int] = None # FeedForward; Also referred to as hidden_dim

    def model_post_init(cls, context):

        if cls.n_local_heads == -1:
           cls.n_local_heads = cls.n_head # MultiHeadedAttention

        if cls.intermediate_size is None:
            # Similar to `Llama2`
            hidden_dim = 4 * cls.n_embd
            n_hidden = int(2 * hidden_dim / 3)
            cls.intermediate_size = _find_multiple(n_hidden, 256)

    @classmethod
    def from_params(cls, file_path: Path) -> "GptModelConfig":

        params = json.loads(file_path.read_text())
        return cls(
            vocab_size=params["vocab_size"],
            n_layer=params["n_layers"],
            n_head=params["n_heads"],
            n_embd=params["dim"],
            n_ctx=2048,
            layer_norm_epsilon=params["norm_eps"]
        )