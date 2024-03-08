import json
from pathlib import Path

import torch

from lego_gpt.blocks.transformer import GptModelConfig
from lego_gpt.blocks.language_modeling import LanguageModeling


class TinyLlamaConfig(GptModelConfig):

    @classmethod
    def from_file(cls, file_path: Path) -> "TinyLlamaConfig":

        config = json.loads(file_path.read_text())
        return cls(
            vocab_size=config["vocab_size"],
            n_layer=config["num_hidden_layers"],
            n_head=config["num_attention_heads"],
            n_local_heads=config["num_key_value_heads"],
            n_embd=config["hidden_size"],
            n_ctx=2048,
            layer_norm_epsilon=config["rms_norm_eps"],
            intermediate_size=config["intermediate_size"]
        )

class TinyLlamaForCausalLM(LanguageModeling):
    """[TinyLlama: An Open-Source Small Language Model]( \
    https://arxiv.org/pdf/2401.02385v1.pdf)"""

    def __init__(self, config: TinyLlamaConfig) -> None:

        super().__init__(config)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, *args) -> None:

        for name in tuple(state_dict):
            if name.startswith("model"):
                if name == "model.embed_tokens.weight":
                    state_dict["model.tok_embeddings.weight"] = state_dict.pop(name)
                elif name == "model.norm.weight":
                    state_dict["norm.weight"] = state_dict.pop(name)
                elif name.endswith("input_layernorm.weight"):
                    new_name = name.replace("input_layernorm", "attention_norm")
                    state_dict[new_name] = state_dict.pop(name)
                elif name.endswith("post_attention_layernorm.weight"):
                    new_name = name.replace("post_attention_layernorm", "ffn_norm")
                    state_dict[new_name] = state_dict.pop(name)
                elif name.startswith("model.layers"): 
                    if name.endswith("q_proj.weight"):
                        prefix = name.replace("q_proj.weight", "")
                        new_prefix = prefix.replace("self_attn", "attention")
                        wq = state_dict.pop(name)
                        wk = state_dict.pop(prefix + "k_proj.weight")
                        wv = state_dict.pop(prefix + "v_proj.weight")
                        state_dict[new_prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
                        state_dict[new_prefix + "wo.weight"] = state_dict.pop(prefix + "o_proj.weight")
                    if name.endswith("down_proj.weight"):
                        prefix = name.replace("down_proj.weight", "")
                        new_prefix = prefix.replace("mlp", "feed_forward")
                        # In PyTorch linear function, the weights are multiplied after a transpose operation
                        state_dict[new_prefix + "w2.weight"] = state_dict.pop(name) # Shape: [hidden_size, intermediate_size]
                        state_dict[new_prefix + "w1.weight"] = state_dict.pop(prefix + "gate_proj.weight") # Shape: [intermediate_size, hidden_size]
                        state_dict[new_prefix + "w3.weight"] = state_dict.pop(prefix + "up_proj.weight") # Shape: [intermediate_size, hidden_size]
