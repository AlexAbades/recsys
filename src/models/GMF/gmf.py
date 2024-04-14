import torch
from torch import Tensor, nn


class GeneralMatrixFactorization(nn.Module):
    """
    Implements the Generalized Matrix Factorization (GMF) model within a PyTorch module.
    The model operates by element-wise multiplication of user and item latent feature vectors,
    which can optionally be followed by a linear transformation to predict a rating or preference score.

    Attributes:
        mf_dim (int): Dimensions of the latent features that represent the original data in a reduced form.
        Which is also the dimension of latent feature vectors. If specified, enables a linear prediction layer.

    Methods:
        forward(mf_user_latent, mf_item_latent): Computes the GMF prediction from user and item latent vectors.
        _init_weight_(): Initializes weights with a normal distribution.
    """

    def __init__(self, mf_dim: int = None) -> None:
        """
        Initializes the GMF model, optionally including a linear prediction
        layer based on the provided feature dimension.

        Parameters:
            mf_dim (int, optional): Dimensions of the latent features, number of factors in Matrix Factorization
        """
        super().__init__()
        self.mf_dim = mf_dim

        if mf_dim:
            self.predict_layer = nn.Linear(mf_dim, 1)

    def forward(self, mf_user_latent: Tensor, mf_item_latent: Tensor) -> Tensor:
        """
        Performs the forward pass of the GMF model.

        Parameters:
            mf_user_latent (Tensor): Latent vector for the user.
            mf_item_latent (Tensor): Latent vector for the item.

        Returns:
            Tensor: The GMF prediction, either as an element-wise product vector or a single prediction value,
            depending on the model configuration.
        """

        mf_vector = torch.mul(mf_user_latent, mf_item_latent)

        if self.mf_dim:
            return self.predict_layer(mf_vector)

        return mf_vector
