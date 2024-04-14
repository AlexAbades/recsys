from typing import List

import torch
from torch import nn

from src.models.Embeding.general_embeddings import GeneralEmbeddings
from src.models.GMF.gmf import GeneralMatrixFactorization
from src.models.MLP.mlp import MultiLayerPerceptron


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(
      self,
      num_users: int,
      num_items: int,
      mf_dim: int,
      layers: List[int],
      binary_classification: bool = True,
    ) -> None:
      """
      Initialize the NCF model.

      Args:
        num_users (int): The number of users in the dataset.
        num_items (int): The number of items in the dataset.
        mf_dim (int): The dimension of the matrix factorization embeddings.
        layers (List[int]): A list of integers representing the sizes of the MLP layers.
        binary_classification (bool, optional): Whether to perform binary classification. Defaults to True.
      """
      super().__init__()
      mlp_in_layer = layers[0]
      mlp_out_layer = layers[-1]
      self.binary_classification = binary_classification

      # Init Embeddings
      self.embeddings = GeneralEmbeddings(num_users, num_items, mf_dim, mlp_in_layer)

      # GMF
      self.GMF_model = GeneralMatrixFactorization()

      # MLP
      self.MLP_model = MultiLayerPerceptron(layers)

      # Prediction layer
      self.predict_layer = nn.Linear(mf_dim + mlp_out_layer, 1)

    def forward(self, user_input, item_input):
        """
        Forward pass of the NCF model.

        Parameters:
            user_input (Tensor): Input tensor containing user IDs.
            item_input (Tensor): Input tensor containing item IDs.

        Returns:
            Tensor: The predicted preferences or ratings, depending on the use case
            (interaction prediction or rating prediction).
        """
        # Embeddings
        mf_user_latent, mf_item_latent, mlp_vector = self.embeddings(
            user_input, item_input
        )

        # GMF
        mf_prediction_vector = self.GMF_model(mf_user_latent, mf_item_latent)

        # MLP
        mlp_prediction_vector = self.MLP_model(mlp_vector)

        # Concat
        prediction_vector = torch.cat(
            [mf_prediction_vector, mlp_prediction_vector], dim=-1
        )

        prediction = self.predict_layer(prediction_vector)

        if self.binary_classification:
            prediction = torch.sigmoid(prediction)
            return prediction

        return prediction
