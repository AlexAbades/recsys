from typing import List

import torch
from torch import nn

from src.models.contextNFC.gmf1 import GMF
from src.models.embedding.embedding import Embedding
from src.models.mlp.mlp import MLP


class DeepNCF(nn.Module):
    """
    Implements the Context Aware Neural Collaborative Filtering (DeepNCF) model, which combines 
    Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP).

    Reference:
        Moshe Unger et al. "Context-Aware Recommendations Based on Deep Learning Frameworks" in ACM 2020.

    Parameters:
        num_users (int): The number of unique users.
        num_items (int): The number of unique items.
        num_context (int): The number of context features.
        mf_dim (int): The dimensionality of the latent features for the GMF part.
        layers (List[int]): The sizes of the layers for the MLP part, including the input layer.

    Attributes:
        embeddings (Embedding): Embedding layer that generates user and item latent vectors.
        GMF_model (GMF): The GMF model part of DeepNCF.
        MLP_model (MLP): The MLP model part of DeepNCF.
        predict_layer (nn.Linear): Final prediction layer that combines features from both GMF and MLP pathways.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_context: int,
        mf_dim: int,
        layers: List[int],
    ) -> None:
        """
        Initializes the DeepNCF model.

        Parameters:
            num_users (int): Number of users.
            num_items (int): Number of items.
            num_context (int): Number of contextual Features
            mf_dim (int): Dimensionality of the latent feature vectors
            layers (List[int]): Layers of the MLP including first and last layers
        """
        super().__init__()
        mlp_in_layer = layers[0]
        mlp_out_layer = layers[-1]

        # Init Embeddings
        self.embeddings = Embedding(
            num_users, num_items, num_context, mf_dim, mlp_in_layer
        )
        # GMF
        self.GMF_model = GMF()
        # MLP
        self.MLP_model = MLP(layers)
        # Prediction layer
        self.predict_layer = nn.Linear(mf_dim + mlp_out_layer, 1)

    def forward(self, user_input, item_input, context_input):
        """
        Forward pass of the DeepNCF model.

        Parameters:
            user_input (Tensor): Input tensor containing user IDs.
            item_input (Tensor): Input tensor containing item IDs.
            context_input (Tensor): Input tensor containing context feature for the user IDs.

        Returns:
            Tensor: The predicted preferences or ratings, depending on the use case 
            (interaction prediction or rating prediction).
        """
        # Embeddings
        mf_user_latent, mf_item_latent, mlp_vector = self.embeddings(
            user_input, item_input, context_input
        )
        # GMF
        mf_prediction_vector = self.GMF_model(mf_user_latent, mf_item_latent)
        # MLP
        mlp_prediction_vector = self.MLP_model(mlp_vector)
        # Concat
        prediction_vector = torch.cat(
            [mf_prediction_vector, mlp_prediction_vector], dim=-1
        )
        # Prediction: sigmoid for interaction, without sigmoid for Rating, Better add this in the model definition
        prediction = torch.sigmoid(self.predict_layer(prediction_vector))

        return prediction