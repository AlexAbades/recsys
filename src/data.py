from src.data.PreProcessData import PreProcessDataNCF
from src.utils.tools.tools import ROOT_PATH

if __name__ == "__main__":
    raw_data_path = "/work3/s212784/data/processed/YELP/data_yelp.csv"
    processed_folder_path = "/work3/s212784/data/processed/YELP/NFC"
    try:
        print("Attempting to PreProcess Data")
        data = PreProcessDataNCF(
            raw_data_path,
            user_column="userId",
            item_column="businessId",
            interaction_column="stars",
        )
        print("Data Initialized. Attempting to train/test:")
        data.split_traintest()
        print("Creating negative samples")
        data.negative_samples(99)
        data.save_data()
    except Exception as e:
        print(e)
