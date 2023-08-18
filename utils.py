import math
import torch
from torch import Tensor
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def bincount_along_dim(x, max_val, dim=-1):
    # bincount along any dimension
    # c.f. https://github.com/pytorch/pytorch/issues/32306
    assert x.dtype is torch.int64, "only integral (int64) tensor is supported"
    cnt = x.new_zeros(x.size(0), max_val)
    # no scalar or broadcasting `src` support yet
    # c.f. https://github.com/pytorch/pytorch/issues/5740
    return cnt.scatter_add_(dim=1, index=x, src=x.new_ones(()).expand_as(x))


def get_returns(rewards):
    returns = torch.cumsum(rewards, dim=0)
    returns = rewards - returns + returns[-1:None]
    return returns
