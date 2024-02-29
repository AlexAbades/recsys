import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import scipy.sparse as sp
from src.models.mlp.mlp import MLP

class NFC(nn.Module):
    def __init__(
        self, num_users, num_items, mf_dim, hidden_dim=[10], reg_mf=0, reg_layers=[0]
    ) -> None:
        super(NFC, self).__init__()
        self.mf_dim = mf_dim
        self.num_layers = len(hidden_dim)

        # MF embeddings
        self.MF_Embedding_User = nn.Embedding(
            num_embeddings=num_users, embedding_dim=self.mf_dim
        )
        self.MF_Embedding_Item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=self.mf_dim
        )

        # MLP
        # MLP - Embeddings
        MLP_in_dim = hidden_dim[0]
        MLP_out_dim = hidden_dim[-1]
        # We divide 1st dim by 2: concatenate feature vectors
        embedding_dim = int(MLP_in_dim / 2)
        self.MLP_Embedding_User = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim
        )
        self.MLP_Embedding_Item = nn.Embedding(
            num_embeddings=num_items, embedding_dim=embedding_dim
        )

        self.MLP_model = MLP(hidden_dim)
        # Last Layer: Prediction layer
        self.predict_layer = nn.Linear(mf_dim + MLP_out_dim, 1)

        self._init_weight_()

    def forward(self, user_input, item_input):
        # MF part
        mf_user_latent = self.MF_Embedding_User(user_input).view(-1, self.mf_dim)
        mf_item_latent = self.MF_Embedding_Item(item_input).view(-1, self.mf_dim)
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)

        # MLP part
        mlp_user_latent = self.MLP_Embedding_User(user_input).view(
            -1, self.MLP_layers[0].in_features
        )
        mlp_item_latent = self.MLP_Embedding_Item(item_input).view(
            -1, self.MLP_layers[0].in_features
        )
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
       
        for layer in self.MLP_layers:
            mlp_vector = F.relu(layer(mlp_vector))
      
        # Concatenate MF and MLP parts
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        # Final prediction layer
        prediction = torch.sigmoid(self.predict_layer(predict_vector))

        return prediction

    def _init_weight_(self):
        nn.init.normal_(self.MF_Embedding_User.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.MF_Embedding_Item.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.MLP_Embedding_User.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.MLP_Embedding_Item.weight, mean=0.0, std=0.01)

        # Optional, not in the paper (works better on deepneural networks)
        for layer in self.MLP_layers:
            nn.init.kaiming_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight)
