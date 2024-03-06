import json
from pathlib import Path

import torch

from lego_gpt.blocks.transformer import GptModelConfig
from lego_gpt.blocks.language_modeling import LanguageModeling


class LlamaConfig(GptModelConfig):

    @classmethod
    def from_file(cls, file_path: Path) -> "LlamaConfig":

        config = json.loads(file_path.read_text())
        return cls(
            vocab_size=config["vocab_size"],
            n_layer=config["n_layers"],
            n_head=config["n_heads"],
            n_embd=config["dim"],
            n_ctx=2048,
            layer_norm_epsilon=config["norm_eps"]
        )

class LlamaForCausalLM(LanguageModeling):

    def __init__(self, config: LlamaConfig) -> None:

        super().__init__(config)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, *args) -> None:

        for name in tuple(state_dict):
            # Match the named parameters in the `LanguageModeling`'s constructor
            if name == "output.weight":
                state_dict["lm_head.weight"] = state_dict.pop(name)
            elif name.startswith("layers") or name.startswith("tok_embeddings"):
                state_dict["model." + name] = state_dict.pop(name)
        
        for name in tuple(state_dict):
            if name.startswith("model.layers") and name.endswith("wq.weight"):
                # Match the named parameters in the `MultiHeadedAttention`'s constructor
                prefix = name.replace("wq.weight", "")
                wq = state_dict.pop(name)
                wk = state_dict.pop(prefix + "wk.weight")
                wv = state_dict.pop(prefix + "wv.weight")
                state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
