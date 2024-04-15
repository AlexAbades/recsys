from typing import Optional

import torch
from torch import Tensor, nn

from src.models.Embeding.general_embeddings import GeneralEmbeddings


class GeneralMatrixFactorization(nn.Module):
    """

    Genral Matrix Factorization (GMF) model.
    Follwoing the "paper" from Simon Funk: https://sifter.org/~simon/journal/20061211.html

    The model can be used with or without own embeddings. If num_users, num_items and mf_dim are provided,
    the embeddings are managed internally. Otherwise, the embeddings are expected to be passed as arguments.

    Prameters:
        num_users (int, optional): The number of users. Defaults to None.
        num_items (int, optional): The number of items. Defaults to None.
        mf_dim (int, optional): The dimension of the matrix factorization. Defaults to None.

    Attributes:
        mf_dim (int): The dimension of the matrix factorization.
        embeddings (GeneralEmbeddings): The embeddings module for user and item inputs. If they are managed internally.
        predict_layer (nn.Linear): The linear layer for prediction. If GMF has to create a prediction.

    Methods:
        forward(user_input, item_input): Performs forward pass of the GMF model.

    """

    def __init__(
        self,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
        mf_dim: int = None,
    ) -> None:
        """
        Initializes the GeneralMatrixFactorization module.

        Prameters:
            num_users (int, optional): The number of users. Defaults to None.
            num_items (int, optional): The number of items. Defaults to None.
            mf_dim (int, optional): The dimension of the matrix factorization. Defaults to None.

        """
        super().__init__()
        self.mf_dim = mf_dim
        self.embeddings = None

        # Initialize embeddings only if all necessary parameters are provided.
        if num_users is not None and num_items is not None and mf_dim is not None:
            self.embeddings = GeneralEmbeddings(num_users, num_items, mf_dim)
            self.predict_layer = nn.Linear(mf_dim, 1)
        # If embeddings are managed externally, expect mf_dim to be set externally for the predict layer
        else:
            if mf_dim is not None:
                self.predict_layer = nn.Linear(mf_dim, 1)

    def forward(self, user_input: Tensor, item_input: Tensor) -> Tensor:
        """
        Performs forward pass of the GMF model.

        Prameters:
            user_input (Tensor): The input tensor for users.
            item_input (Tensor): The input tensor for items.

        Returns:
            Tensor: The output tensor of the GMF model.

        """
        if self.embeddings:
            # Extract embeddings from internal embeddings module
            mf_user_latent, mf_item_latent = self.embeddings(user_input, item_input)
        else:
            # Assume mf_user_latent and mf_item_latent are passed directly as arguments
            mf_user_latent = user_input
            mf_item_latent = item_input

        # GMF computation
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)

        if hasattr(self, "predict_layer"):
            return torch.sigmoid(self.predict_layer(mf_vector))

        return mf_vector
