import argparse
import os

from src.models.PopularItem.popularItem import PopularItem
from src.utils.model_stats.stats import save_accuracy
from src.utils.tools.tools import ROOT_PATH, create_checkpoint_folder
from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    # name
    parser.add_argument("--name", type=str, default="PopularItem")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoints/PopularItem",
        type=str,
        metavar="PATH",
        help="checkpoint directory",
    )
    parser.add_argument("--K", type=int, default=10, help="Number of popular items")
    opts = parser.parse_args()
    return opts


def main(opts):
    # create checkpoint
    args = edict({"processed_data_root": opts.data, "foldername": opts.name})
    data_name, check_point_path = create_checkpoint_folder(args, opts)
    data_abs_path = os.path.join('/home/alexabades/recsys', opts.data)
    print(f"Data path: {data_abs_path}, data name {data_name}")
    popular = PopularItem(
        path=data_abs_path,
        data_name=data_name,
        K=opts.K,
        user_column="",
        item_column="item",
        rating_column="rating",
    )
    save_accuracy(
        chk_path=check_point_path,
        hr=popular.hit_ratio,
        ndcg=popular.ndcg_ratio,
        mrr=popular.mrr,
    )


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
