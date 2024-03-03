import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    
    def __init__(self, n_embd: int, eps: float = 1e-5):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd)) # Learnable scale parameter
        self.eps = eps

    def _norm(self, x):
        
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
