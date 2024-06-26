import torch
from torch import nn


class GeneralEmbeddings(nn.Module):
    def __init__(
        self, num_users, num_items, mf_dim: int = 0, mlp_layer: int = 0
    ) -> None:
        """
        Initialize the Embedding model.

        Parameters:
            num_users (int): The number of users.
            num_items (int): The number of items.
            mf_dim (int, optional): The dimension of the matrix factorization embeddings. Defaults to 0.
            mlp_layer (int, optional): The number of layers in the MLP embeddings. Defaults to 0.
        """
        super().__init__()
        self.mf_dim = mf_dim
        self.mlp_layer = mlp_layer

        # Initialize MF embeddings if mf_dim is specified
        if self.mf_dim > 0:
            self.MF_Embedding_User = nn.Embedding(
                num_embeddings=num_users, embedding_dim=self.mf_dim
            )
            self.MF_Embedding_Item = nn.Embedding(
                num_embeddings=num_items, embedding_dim=self.mf_dim
            )

        # Initialize MLP embeddings if mlp_layer is specified
        if self.mlp_layer > 0:
            self.embedding_dim = int(
                mlp_layer / 2
            )  # assuming a structure for MLP layers
            self.MLP_Embedding_User = nn.Embedding(
                num_embeddings=num_users, embedding_dim=self.embedding_dim
            )
            self.MLP_Embedding_Item = nn.Embedding(
                num_embeddings=num_items, embedding_dim=self.embedding_dim
            )

        # Initialize weights separately if needed
        self._init_weight_()

    def forward(self, user_input, item_input):
        """
        Forward pass of the embedding model.

        Args:
            user_input (torch.Tensor): Tensor containing user input.
            item_input (torch.Tensor): Tensor containing item input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the user latent vector for matrix factorization (MF),
            the item latent vector for matrix factorization (MF), and the concatenated vector for the multi-layer perceptron (MLP).
        """
        outputs = []

        # Matrix factorization embeddings
        if self.mf_dim > 0:
            mf_user_latent = self.MF_Embedding_User(user_input).view(-1, self.mf_dim)
            mf_item_latent = self.MF_Embedding_Item(item_input).view(-1, self.mf_dim)
            outputs.extend([mf_user_latent, mf_item_latent])

        # MLP embeddings
        if self.mlp_layer > 0:
            mlp_user_latent = self.MLP_Embedding_User(user_input).view(-1, self.embedding_dim)
            mlp_item_latent = self.MLP_Embedding_Item(item_input).view(-1, self.embedding_dim)
            mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
            outputs.append(mlp_vector)

        return tuple(outputs)

    def _init_weight_(self):
        """
        Initializes the weights of the user and item embeddings using a normal distribution.

        Args:
            None

        Returns:
            None
        """
        if hasattr(self, "MF_Embedding_User"):
            nn.init.normal_(self.MF_Embedding_User.weight, mean=0.0, std=0.01)
        if hasattr(self, "MF_Embedding_Item"):
            nn.init.normal_(self.MF_Embedding_Item.weight, mean=0.0, std=0.01)
