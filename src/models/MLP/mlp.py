import torch
from torch import nn
from typing import Callable, List, Optional


class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    Args:
        hidden_dims (int or List[int]): The dimensions of the hidden layers. If an int is provided, a single hidden layer
            with that dimension will be created. If a list of ints is provided, multiple hidden layers will be created
            with dimensions specified in the list. Default is None.
        dropout (float): The dropout probability. Default is 0.
        activation (Callable[..., nn.Module]): The activation function to use. Default is nn.ReLU.

    Attributes:
        mlp_model (nn.Sequential): The sequential model representing the MLP.

    """

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
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.mlp_model(x)
