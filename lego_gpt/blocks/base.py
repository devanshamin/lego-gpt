import json
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from pydantic import BaseModel

from lego_gpt import blocks
from lego_gpt.train.utils import get_optimizer
from lego_gpt.train.config import TrainerConfig


class BaseLanguageModelConfig(BaseModel):
    architecture: str # Name of the available model class
    vocab_size: int
    n_ctx: int # Also referred to as block_size or seq_len
    n_embd: int # Also referred to as embed_dim or dim
    n_layer: int


class BaseLanguageModel(nn.Module):

    def get_num_params(self, non_embedding: bool = True):

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # The token embeddings are excluded here because they are used
            # as weights in the final layer
            n_params -= self.wpe.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(
        cls, 
        checkpoint_dir: str, 
        device: torch.device,
        trainer_config: Optional[TrainerConfig] = None,
        **model_kwargs: Dict
    ):

        checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.exists(), "Checkpoint directory not found! Please provide a valid path."
            
        model_files = torch.load(checkpoint_dir / "model.pt", map_location=device)
        config = json.loads((checkpoint_dir / "config.json").read_text())
        
        model_name = config["model_config"]["architecture"]
        model_config_cls = getattr(blocks, f"{model_name}Config", None)
        assert model_config_cls is not None, f"Model config not found for model '{model_name}'"
        config["model_config"].update(model_kwargs)
        model = cls(config=model_config_cls(**config["model_config"]))
        model.load_state_dict(state_dict=model_files["model"])
        model = model.to(device)
        
        if trainer_config:
            trainer_config.update(config["trainer_config"])
        else:
            trainer_config = TrainerConfig(**config["trainer_config"])
        optimizer = get_optimizer(model, config=trainer_config.optimizer_config, device=device)
        optimizer.load_state_dict(state_dict=model_files["optimizer"])

        print(f"No. of model parameters: {model.get_num_params(False):,}")

        return model, optimizer