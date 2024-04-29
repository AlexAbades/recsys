import os

import holidays
import numpy as np
import pandas as pd


def is_holiday(row):
    ca_provinces = [
        "AB",
        "BC",
        "MB",
        "NB",
        "NL",
        "NS",
        "NT",
        "NU",
        "ON",
        "PE",
        "QC",
        "SK",
        "YT",
    ]
    date, state = row["date"], row["state"]
    if state in ca_provinces:
        holiday_list = holidays.Canada(prov=state)
    else:  # For US states and other unspecified codes
        holiday_list = holidays.UnitedStates(
            state=state if state in holidays.US.subdivisions else None
        )
    return date in holiday_list


if __name__ == "__main__":

    file_name = "yelp_data.csv"
    data_raw_folder = "/work3/s212784/data/YELP/"
    data_processed_folder = "/work3/s212784/data/processed/YELP/"

    # Load datasets
    df_review = pd.read_csv(data_raw_folder + "yelp_academic_dataset_review.csv")
    df_user = pd.read_csv(data_raw_folder + "yelp_academic_dataset_user.csv")
    df_business = pd.read_csv(data_raw_folder + "yelp_academic_dataset_business.csv")
    print("Data Loaded")
    # Context of the user
    user_cols = ["user_id", "elite", "yelping_since", "friends"]
    # Location
    business_col = ["business_id", "latitude", "longitude", "state"]
    # Review
    review_cols = ["user_id", "business_id", "date", "stars"]
    # Merge
    merged_df = pd.merge(
        df_review[review_cols],
        df_user[user_cols],
        on="user_id",
        how="inner",
        suffixes=["_review", "_user"],
    )
    print("First Merge done")

    del df_review, df_user
    print("Variables deleted")

    # Merge
    data = pd.merge(merged_df, df_business[business_col], on="business_id", how="inner")
    print("Second Merge done")
    del df_business
    print("Variables deleted")

    # Transform to date
    data["date"] = pd.to_datetime(data["date"])
    data["yelping_since"] = pd.to_datetime(data["yelping_since"])
    print("Date Transformed")

    # Classify workday/weekend
    # data["isweekend"] = data["date"].dt.dayofweek.apply(classify_day)
    data["isweekend"] = data["date"].dt.dayofweek >= 5
    print("Weekend/worrkday done")

    # Classify daytime
    # data["daytime"] = data["date"].apply(lambda x: classify_time_of_day(x)).astype('category')
    conditions = [
        (data["date"].dt.hour >= 5) & (data["date"].dt.hour < 7),
        (data["date"].dt.hour >= 7) & (data["date"].dt.hour < 12),
        (data["date"].dt.hour >= 12) & (data["date"].dt.hour < 14),
        (data["date"].dt.hour >= 14) & (data["date"].dt.hour < 17),
        (data["date"].dt.hour >= 17) & (data["date"].dt.hour < 20),
    ]
    choices = [
        "sunrise",
        "morning",
        "noon",
        "afternoon",
        "evening",
    ]
    data["daytime"] = np.select(conditions, choices, default="night")
    data["daytime"] = data["daytime"].astype("category")
    data["hour"] = data["date"].dt.hour
    print("Daytime done")

    # Obtain weeknumber
    data["week_number"] = data["date"].dt.isocalendar().week
    print("week_number done")

    # Obtain Holidays
    data["isHoliday"] = data.apply(is_holiday, axis=1)
    print("isHoliday done")

    # Transofrm number Friends + elitegit a
    data["num_elite"] = data.elite.apply(
        lambda x: len(x.split(",")) if not pd.isna(x) else 0
    )
    data["num_firends"] = data.friends.apply(
        lambda x: len(x.split(",")) if not pd.isna(x) else 0
    )
    print("num_elite & num_firends done")

    # Obtain season
    years = data["date"].dt.year
    # Define start dates for each season in a given year
    spring_starts = pd.to_datetime(years.astype(str) + "-03-20")
    summer_starts = pd.to_datetime(years.astype(str) + "-06-21")
    fall_starts = pd.to_datetime(years.astype(str) + "-09-22")
    winter_starts = pd.to_datetime(years.astype(str) + "-12-21")
    # Conditions for each season
    conditions = [
        (data["date"] >= spring_starts) & (data["date"] < summer_starts),
        (data["date"] >= summer_starts) & (data["date"] < fall_starts),
        (data["date"] >= fall_starts) & (data["date"] < winter_starts),
        (data["date"] >= winter_starts) | (data["date"] < spring_starts),
    ]
    # Season names
    choices = ["spring", "summer", "fall", "winter"]
    # Apply conditions and choices
    data["season"] = np.select(conditions, choices, default="winter")
    data["season"] = data["season"].astype("category")
    print("season done")

    # Obtain Seniority on app
    max_date = data.date.max().year
    data["seniority"] = data["yelping_since"].apply(lambda x: max_date - x.year)
    print("seniority done")

    # Transform uuid4
    data["userId"] = pd.factorize(data["user_id"])[0]
    data["businessId"] = pd.factorize(data["business_id"])[0]
    print("Ids Transformation done")

    # Filter desired columns
    columns = [
        "userId",
        "businessId",
        "stars",
        "latitude",
        "longitude",
        "isweekend",
        "daytime",
        "hour",
        "week_number",
        "season",
        "isHoliday",
        "num_firends",
        "num_elite",
        "seniority",
    ]

    # Save csv
    os.makedirs(data_processed_folder, exist_ok=True)
    file_name = "yelp_data_V2.csv"
    file_path = os.path.join(data_processed_folder, file_name)
    print(f"Attempting to save in directory: {data_processed_folder}")

    try:
        data[columns].to_csv(
            data_processed_folder + file_name, sep="\t", header=False, index=False
        )
        print(f"Saved into {data_processed_folder}, Filename: {file_name}")

        readme_path = os.path.join(data_processed_folder, "ReadMe.txt")
        with open(readme_path, "w") as f:
            f.write(f"Summary of data {file_name}\n")
            f.write(data.dtypes.to_string())  # Convert dtype Series to string
    except Exception as e:
        print(f"The following Problem occured {e}")
