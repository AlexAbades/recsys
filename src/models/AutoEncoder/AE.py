import warnings
from typing import Callable, List

from torch import nn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int] = None,
        dropout: float = 0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        init_weights: bool = False,
    ) -> None:
        """
        Autoencoder following an hourglass structure

        Parameters:
          - hidden_dims (List[int]): List indicating the hidden dimensions of the Encoder
          - dropout (float): Dropout for all hidden layers
          - activation (Callable): Activation function, defaults to ReLU
          - init_weights (bool): Whether to initialize the weights, defaults to False
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("AutoEncoder initialized without dimensions")
        if len(hidden_dims) < 2:
            warnings.warn("AutoEncoder with a single hidden layer.")

        # Build the encoder layers
        encoder_layers = []
        for i in range(len(hidden_dims) - 1):
            encoder_layers.extend(
                [
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    activation(),
                    nn.Dropout(dropout),
                ]
            )

        # Build the decoder layers (reverse structure of the encoder)
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]  # Reverse the dimensions for the decoder
        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend(
                [
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    activation(),
                    nn.Dropout(dropout),
                ]
            )

        # Removing the last Dropout from the encoder and decoder
        encoder_layers = encoder_layers[:-1]
        decoder_layers = decoder_layers[:-1]

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers, nn.Sigmoid())

        if init_weights:
            self._init_weights()

    def forward(self, x):
        """
        Forward pass of the AutoEncoder

        Parameters:
          - x: Input tensor

        Returns:
          - dict: Dictionary containing the latent representation and the reconstructed input
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return {"latent": z, "prediction": x_hat}

    def _init_weights(self):
        """
        Initialize the weights of the linear layers in the AutoEncoder
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
