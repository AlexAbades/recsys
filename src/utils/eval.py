import math
from typing import List


def getHR(ranklist: List[int], gtItems: List[int]) -> int:
    """
    Evaluates if the ground truth item/s (gtItems) is/are in the ranked list (ranklist).

    Parameters:
    - ranklist (List[int]): List of ranked item IDs.
    - gtItem (List[int]): ID/s of the ground truth item/s to find.

    Returns:
    - int: 1 if gtItem is found, 0 otherwise.
    """
    for item in ranklist:
        if item in gtItems:
            return 1
    return 0


def getRR(ranklist: List[int], gtItems: List[int]) -> float | int:
    """
    Evaluates the Reciprocal Rank of a relevant item in the ranklist

    Parameters:
    - ranklist (List[int]): List of ranked item IDs.
    - gtItem (List[int]): ID/s of the ground truth item/s to find.

    Returns:
    - float | int: 1/relK_i if gtItem is found in ranklist, 0 otherwise.
    """
    for item in ranklist:
        if item not in gtItems:
            return 0
        return 1 / (gtItems.index(item) + 1)
    return


def getBinaryDCG(ranklist: List[int], gt_items: List[int]):
    """
    Parameters:
    - ranklist (List[int]): List of ranked item IDs.
    - gtItem (List[int]): ID of the ground truth item to find.

    Returns:
      - dcg (float): The DCG value, higher indicates better ranking of relevant items.
    """
    dcg = 0
    for idx, item_id in enumerate(ranklist):
        if item_id in gt_items:
            dcg += (1) / (math.log2(idx + 2))
    return dcg


def getBinaryIDCG(gt_items: List[int]):
    """
    Calculates the Ideal Discounted Cumulative Gain (IDCG) for a list of ground truth items

    Parameters:
      - gt_items (List[int]): gtItem (List[int]): ID of the ground truth item to find.

    Returns:
      - idcg (float): The IDCG value, representing the optimal ranking score for the given relevance scores.
    """

    idcg = 0
    for idx, _ in enumerate(gt_items):
        idcg += (1) / (math.log2(idx + 2))
    return idcg


def getLinearDCG(ranklist_scores: List[int | float]) -> float:
    """
    Calculates the Discounted Cumulative Gain (DCG) for a list of item relevance scores.

    Parameters:
    - ranklist_scores (List[int | float]): Relevance scores for predicted ranked items.

    Returns:
    - dcg (float): The DCG value, higher indicates better ranking of relevant items.
    """
    dcg = 0
    for idx, rel in enumerate(ranklist_scores):
        dcg += (rel) / math.log2(idx + 2)
    return dcg


def getExponentialDCG(ranklist_scores: List[int | float]) -> float:
    """
    Calculates the Exponential Discounted Cumulative Gain (DCG) for a list of item relevance scores.

    Parameters:
    - ranklist_scores (List[int | float]): Relevance scores for predicted ranked items.

    Returns:
    - dcg (float): The DCG value, higher indicates better ranking of relevant items.
    """
    dcg = 0
    for idx, rel in enumerate(ranklist_scores):
        dcg += (2**rel) / math.log2(idx + 2)
    return dcg


def getExponentialIDCG(gt_scores: List[int | float]) -> float:
    """
    Calculates the Ideal Exponential Discounted Cumulative Gain (IDCG) for a list of ground truth scores.

    Parameters:
    - gt_scores (List[int | float]): A list of ground truth scores for items, where scores indicate relevance.

    Returns:
    - float: The IDCG value, indicating the maximum possible DCG based on the given ground truth scores.
    """
    gt_scores_sorted = sorted(gt_scores, reverse=True)
    idcg = 0
    for idx, rel in enumerate(gt_scores_sorted):
        idcg += (2**rel) / (math.log2(idx + 2))
    return idcg


def getLinearIDCG(gt_scores: List[int | float]) -> float:
    """
    Calculates the Ideal Discounted Cumulative Gain (IDCG) for a list of ground truth scores.

    Parameters:
    - gt_scores (List[int | float]): A list of ground truth scores for items, where scores indicate relevance.

    Returns:
    - float: The IDCG value, indicating the maximum possible DCG based on the given ground truth scores.
    """
    gt_scores_sorted = sorted(gt_scores, reverse=True)
    idcg = 0
    for idx, rel in enumerate(gt_scores_sorted):
        idcg += (rel) / (math.log2(idx + 2))
    return idcg
