import torch 
from torch import nn

class Embedding(nn.Module):
  def __init__(self, mf_dim:int=0, mlp_layer:int=0, **embeddings) -> None:
    super().__init__()
    self.mf_dim = mf_dim
    self.mlp_layer = mlp_layer


    for key, value in embeddings.items():
            if mf_dim:
                setattr(self, key, value)  # This correctly assigns the value to an attribute named after 'key'

    # MF embeddings
    self.MF_Embedding_User = nn.Embedding(
        num_embeddings=num_users, embedding_dim=self.mf_dim
    )
    self.MF_Embedding_Item = nn.Embedding(
        num_embeddings=num_items, embedding_dim=self.mf_dim
    )

    # MLP
    # MLP - Embeddings
    MLP_in_dim = mlp_layer

    # MLP_out_dim = layers[-1]  # 64
    # We divide 1st dim by 2: concatenate feature vectors
    embedding_dim = int(MLP_in_dim / 2)  
    self.MLP_Embedding_User = nn.Embedding(
        num_embeddings=num_users, embedding_dim=embedding_dim
    )
    self.MLP_Embedding_Item = nn.Embedding(
        num_embeddings=num_items, embedding_dim=embedding_dim
    )
