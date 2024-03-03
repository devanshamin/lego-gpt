from typing import Optional

from pydantic import BaseModel
from transformers.trainer_utils import SchedulerType


class OptimizerConfig(BaseModel):
    optim: str = "AdamW8bit" # `torch.optim` or `bnb.optim`
    weight_decay: float = 1e-1 
    lr: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95

class TrainerConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 2
    eval_interval: int = 2
    eval_iters: int = 200
    gradient_accumulation_steps: int = 1#5 * 8
    grad_clip: float = 1.0 # Helps with mitigating exploding gradients
    lr_scheduler: SchedulerType = SchedulerType.COSINE # LambdaLR
    lr_warmup_steps: int = 100 # LambdaLR
    optimizer_config: OptimizerConfig = OptimizerConfig()
    enable_amp: bool = False # Automatic mixed precision for device == 'cuda'

    num_training_steps: Optional[int] = None # LambdaLR

    def set_num_training_steps(self, steps_per_epoch : int) -> None:

        self.num_training_steps = self.epochs * steps_per_epoch