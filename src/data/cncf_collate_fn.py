import torch

def cncf_collate_negative_sampling(batch):
    """
    Collate function for negative sampling.

    Args:
        batch (list): A list of dictionaries representing the batch of data samples. Each dictionary contains the following keys:
            - "user": Tensor of shape [batch_size, ...] representing the user data.
            - "item": Tensor of shape [batch_size, ...] representing the item data.
            - "context": Tensor of shape [batch_size, ..., context_size] representing the context data.
            - "rating": Tensor of shape [batch_size, ...] representing the rating data.

    Returns:
        dict: A dictionary containing the collated batch data. The dictionary has the following keys:
            - "user": Tensor of shape [batch_size*N, ...] representing the user data, where N is the number of negative samples.
            - "item": Tensor of shape [batch_size*N, ...] representing the item data, where N is the number of negative samples.
            - "context": Tensor of shape [batch_size*N, ..., context_size] representing the context data, where N is the number of negative samples.
            - "rating": Tensor of shape [batch_size*N, ...] representing the rating data, where N is the number of negative samples.
    """
    batch_user = torch.stack([item["user"] for item in batch], dim=0)
    batch_item = torch.stack([item["item"] for item in batch], dim=0)
    batch_context = torch.stack([item["context"] for item in batch], dim=0)
    batch_rating = torch.stack([item["rating"] for item in batch], dim=0)
    batch_gtItem = torch.stack([item["gtItem"] for item in batch], dim=0)

    # Flatten the batch dimensions to have shape [batch_size*N, ...] since each item contains a positive and N negative samples
    batch_user = batch_user.view(-1)
    batch_item = batch_item.view(-1)
    batch_context = batch_context.view(-1, batch_context.size(-1))
    batch_rating = batch_rating.view(-1)
    batch_gtItem = batch_gtItem.view(-1)

    return {
        "user": batch_user,
        "item": batch_item,
        "context": batch_context,
        "rating": batch_rating,
        "gtItem": batch_gtItem,
    }