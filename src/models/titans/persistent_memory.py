import torch
import torch.nn as nn


class PersistentMemory(nn.Module):
    """
    Learnable but data-independent tokens prepended to each chunk.
    Encodes task-level knowledge. Fixed at test time.
    """

    def __init__(self, d_model: int, n_tokens: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

    def forward(self, B: int) -> torch.Tensor:
        return self.tokens.expand(B, -1, -1)  # (B, N_p, D)
