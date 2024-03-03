import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, n_embd: int, intermediate_size: int):
        
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate_size, bias=False)
        self.w3 = nn.Linear(n_embd, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, n_embd, bias=False)

    def forward(self, x: torch.FloatTensor):
        
        return self.w2(F.silu(self.w1(x)) * self.w3(x))