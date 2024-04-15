from torch.utils.data import DataLoader

from src.data.cncf_collate_fn import ncf_negative_sampling
from src.data.nfc_dataset import NCFDataset
from src.models.NCF.nfc import NeuralCollaborativeFiltering
import torch
from torch import nn, optim

from src.utils.tools.tools import ROOT_PATH

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_data_path = ROOT_PATH + "/../data/processed/FRAPPE/NCF/frappe_ncf/"
num_users = 654
num_items = 1127
batch_size = 64
layers = [64, 32, 16, 8]
num_factors = 8
_optimizers = {"adam": optim.Adam, "SGD": optim.SGD}
_loss_fn = {"BCE": nn.BCELoss(), "MSE": nn.MSELoss()}
loss_fn = _loss_fn["BCE"]
lr = 0.01

if __name__ == "__main__":
    train_data = NCFDataset(
        processed_data_path,
        split="train",
        n_items=num_items,
        num_negative_samples=5,
    )

    test_data = NCFDataset(
        processed_data_path,
        split="test",
        n_items=num_items,
        num_negative_samples=99,
    )

    # Dataloader
    train_loader = DataLoader(train_data, batch_size, collate_fn=ncf_negative_sampling)
    test_loader = DataLoader(test_data, batch_size, collate_fn=ncf_negative_sampling)

    model = NeuralCollaborativeFiltering(
        num_users=num_users, num_items=num_items, mf_dim=num_factors, layers=layers
    ).to(_device)
    optimizer = _optimizers["adam"](model.parameters(), lr=lr)

    # For loop to see the batch and the index that batch is getting
    for i, batch in enumerate(train_loader):
        
        user = batch["user"].to(_device)
        item = batch["item"].to(_device)
        rating = batch["rating"].to(_device)
        rating = rating.view(-1, 1)
        gtItem = batch["gtItem"].to(_device)

        predictions = model(user, item)

        loss = loss_fn(predictions, rating)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

