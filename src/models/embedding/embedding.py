import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self, num_users, num_items, num_context, mf_dim: int = 0, mlp_layer: int = 0
    ) -> None:
        super().__init__()
        self.mf_dim = mf_dim
        self.mlp_layer = mlp_layer

        # MF
        self.MF_Embedding_User = nn.Embedding(
            num_embeddings=num_users, embedding_dim=self.mf_dim
        )
        self.MF_Embedding_Item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=self.mf_dim
        )

        # MLP
        self.embedding_dim = int((mlp_layer - num_context) / 2)
        self.MLP_Embedding_User = nn.Embedding(
            num_embeddings=num_users, embedding_dim=self.embedding_dim
        )
        self.MLP_Embedding_Item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=self.embedding_dim
        )
        self._init_weight_()

    def forward(self, user_input, item_input, context_input):
        # MF
        mf_user_latent = self.MF_Embedding_User(user_input).view(-1, self.mf_dim)
        mf_item_latent = self.MF_Embedding_Item(item_input).view(-1, self.mf_dim)

        # MLP
        mlp_user_latent = self.MLP_Embedding_User(user_input).view(
            -1, int(self.embedding_dim)
        )
        mlp_item_latent = self.MLP_Embedding_Item(item_input).view(
            -1, int(self.embedding_dim)
        )

        mlp_vector = torch.cat(
            [
                mlp_user_latent,
                mlp_item_latent,
                context_input,
            ],
            dim=-1,
        )

        return mf_user_latent, mf_item_latent, mlp_vector
    
    def _init_weight_(self):
        """
        Initializes the weights of the user and item embedding layers with normal distribution.
        """
        nn.init.normal_(self.MF_Embedding_User.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.MF_Embedding_Item.weight, mean=0.0, std=0.01)
