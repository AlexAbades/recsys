import math
import heapq  # for retrieval topK
import multiprocessing
from typing import List
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(model, device, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _device
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _device = device

    hits, ndcgs = [], []

    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)

    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    ground_truth_item = rating[1]
    items.append(ground_truth_item)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype="int32")

    # Model set to evaluation
    _model.eval()

    # Convert to tensors
    users_tensor = torch.tensor(users, dtype=torch.long)
    items_tensor = torch.tensor(items, dtype=torch.long)

    # Create a dataset and dataloader for batching
    dataset = TensorDataset(users_tensor, items_tensor)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            user_batch = batch[0].to(_device)
            item_batch = batch[1].to(_device)
            batch_predictions = _model(user_batch, item_batch)
            predictions.extend(batch_predictions.cpu().numpy())

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)  # Just a List
    hr = getHitRatio(ranklist, ground_truth_item)
    ndcg = getNDCG(ranklist, ground_truth_item)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist: List, gtItem):
    """
    Args:
      - ranklist (List): The list of items, ranked according to some criteria.
      - gtItem: The ground truth item for which the NDCG score is to be calculated.
    Returns:
      - A floating-point number representing the NDCG score
    """
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0
