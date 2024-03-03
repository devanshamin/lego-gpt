import inspect

import torch
import bitsandbytes as bnb

from .config import OptimizerConfig


def get_optimizer(model, config: OptimizerConfig, device: torch.device):

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay}, # matmuls & embeddings
        {'params': nodecay_params, 'weight_decay': 0.0} # biases, layernorms
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"No. of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"No. of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optim_cls = getattr(torch.optim, config.optim, None) or getattr(bnb.optim, config.optim, None)
    assert optim_cls is not None, \
        f"{config.optim} not found in PyTorch and Bitsandbytes optimizers! Provide a valid optimizer."

    extra_args = {}
    if (
        "fused" in inspect.signature(optim_cls).parameters
        and device == "cuda"
    ):
        extra_args["fused"] = True

    optimizer = optim_cls(
        optim_groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        **extra_args
    )
    return optimizer