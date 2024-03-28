from src.models.mlp.mlp import MLP

if __name__ == "__main__":
  print("hello")
  # The in dimensions is the concatenation of user and item length or size, in this case both have same length
  EMBEDDING_SIZE = 3
  N_USERS = 5
  # TODO: Understand Why N_ITEMS is the output size
  N_ITEMS = 10
  a = MLP(2*EMBEDDING_SIZE, N_ITEMS, [6, 4, 2], dropout=0)
  print(a)