import torch

from lego_gpt.blocks.base import BaseLanguageModelConfig
from lego_gpt.blocks.language_modeling import LanguageModeling


class LlamaForCausalLM(LanguageModeling):

    def __init__(self, config: BaseLanguageModelConfig):

        super().__init__(config)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, *args):

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
