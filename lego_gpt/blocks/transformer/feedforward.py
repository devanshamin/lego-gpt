import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Feed Forward Network (FNN) [SwiGLU](https://arxiv.org/pdf/2002.05202.pdf) layer.
    
    `FFNSwiGLU (x, W, V, W2) = (Swi(xW) ⊙ xV) W2`
    * Swi - [Swish](https://arxiv.org/abs/1710.05941v1) activation function
    * GLU - GLU (Gated Linear Units) is a neural network layer
    """

    def __init__(self, n_embd: int, intermediate_size: int):
        
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate_size, bias=False)
        self.w3 = nn.Linear(n_embd, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, n_embd, bias=False)

    def forward(self, x: torch.FloatTensor):
        
        # `F.silu` - acts as gating mechanism that controls the flow of information
        #            from the linear transformation
        # `F.silu(self.w1(x)) * self.w3(x)` - GLU; linear transformation followed 
        #            by a gating mechanism
        return self.w2(F.silu(self.w1(x)) * self.w3(x))