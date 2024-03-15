import torch
from torch import nn
from typing import Callable, List, Optional


class MLP(nn.Module):

    def __init__(
        self,
        hidden_dims: int | List[int] = None,
        dropout: float = 0,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        super().__init__()

        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for layer in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[layer], hidden_dims[layer+1]))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            
        self.mlp_model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_model(x)
