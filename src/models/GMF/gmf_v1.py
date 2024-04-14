import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim) -> None:
        super().__init__()
        self.mf_dim = mf_dim


        # MF embeddings
        self.MF_Embedding_User = nn.Embedding(
            num_embeddings=num_users, embedding_dim=self.mf_dim
        )
        self.MF_Embedding_Item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=self.mf_dim
        )

        self.predict_layer = nn.Linear(mf_dim, 1)
        self._init_weight_()

    def forward(self, user_input, item_input):

        # MF part
        mf_user_latent = self.MF_Embedding_User(user_input).view(-1, self.mf_dim)
        mf_item_latent = self.MF_Embedding_Item(item_input).view(-1, self.mf_dim)
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)

        prediction = torch.sigmoid(self.predict_layer(mf_vector))

        return prediction

    def _init_weight_(self):
        nn.init.normal_(self.MF_Embedding_User.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.MF_Embedding_Item.weight, mean=0.0, std=0.01)
