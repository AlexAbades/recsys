import os
from datetime import datetime

import holidays
import numpy as np
import pandas as pd

from src.utils.tools.tools import ROOT_PATH


def classify_day(day):
    if day >= 5:
        return "weekend"
    else:
        return "workday"


def classify_time_of_day(time):
    if 5 <= time.hour < 7:
        return "sunrise"
    elif 7 <= time.hour < 12:
        return "morning"
    elif 12 <= time.hour < 14:
        return "noon"
    elif 14 <= time.hour < 17:
        return "afternoon"
    elif 17 <= time.hour < 20:
        return "evening"
    else:
        return "night"


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
    return "isholiday" if date in holiday_list else "notholiday"


def get_season(date):
    # Ensure the input is a datetime object
    if not isinstance(date, datetime):
        raise ValueError("The date must be a datetime object")

    spring_start = datetime(date.year, 3, 20)
    summer_start = datetime(date.year, 6, 21)
    fall_start = datetime(date.year, 9, 22)
    winter_start = datetime(date.year, 12, 21)

    if date >= spring_start and date < summer_start:
        return "spring"
    elif date >= summer_start and date < fall_start:
        return "summer"
    elif date >= fall_start and date < winter_start:
        return "fall"
    else:
        return "winter"


if __name__ == "__main__":

    file_name = "data_yelp.csv"
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
    data["isweekend"] = data["date"].dt.dayofweek.apply(classify_day)
    print("Weekend/worrkday done")
    # Classify daytime
    data["daytime"] = data["date"].apply(lambda x: classify_time_of_day(x))
    print("Daytime done")
    # Obtain weeknumber
    data["week_number"] = data["date"].dt.isocalendar().week
    print("week_number done")
    # Obtain Holidays
    data["holiday_status"] = data.apply(is_holiday, axis=1)
    print("holiday_status done")
    # Transofrm number Friends + elitegit a
    data["num_elite"] = data.elite.apply(
        lambda x: len(x.split(",")) if not pd.isna(x) else 0
    )
    data["num_firends"] = data.friends.apply(
        lambda x: len(x.split(",")) if not pd.isna(x) else 0
    )
    print("num_elite & num_firends done")
    # Obtain season
    data["season"] = data["date"].apply(get_season)
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
        "week_number",
        "season",
        "holiday_status",
        "num_firends",
        "num_elite",
        "seniority",
    ]
    data[columns]
    # Save csv
    os.makedirs(data_processed_folder, exist_ok=True)
    print(f"Attempting to save in directory: {data_processed_folder}")
    try:
        data.to_csv(data_processed_folder + file_name)
        print(f"saved into {data_processed_folder, file_name}")
    except Exception as e:
        print(f"The following Problem occured {e}")
