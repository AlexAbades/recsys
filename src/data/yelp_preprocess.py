from src.data.nfc_preprocess import PreProcessDataNCF
from src.data.cncf_preprocess import PreProcessDataNCFContextual
from src.utils.tools.tools import ROOT_PATH

if __name__ == "__main__":
    path = "/work3/s212784/data/processed/YELP"
    data_file = "yelp.csv"
    user_column = "userId"
    item_column = "businessId"
    ratings_column = "stars"
    ctx_categorical_columns = ["isweekend", "season", "isHoliday", "daytime"]
    ctx_numerical_columns = [
        "latitude",
        "longitude",
        "week_number",
        "num_firends",
        "num_elite",
        "seniority",
    ]
    columns_to_transform = {"cyclical": ["week_number"]}
    min_interactions = 2
    try:
        print("Attempting to PreProcess Data")
        data = PreProcessDataNCFContextual(
            path=path,
            data_file=data_file,
            user_column=user_column,
            item_column=item_column,
            rating_column=ratings_column,
            ctx_categorical_columns=ctx_categorical_columns,
            ctx_numerical_columns=ctx_numerical_columns,
            columns_to_transform=columns_to_transform,
            min_interactions=min_interactions,
        )
        data.save_data(folder_name="yelp_2.1_ctx")
    except Exception as e:
        print(e)
