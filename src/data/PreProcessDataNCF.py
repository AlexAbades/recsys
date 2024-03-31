
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


class PreProcessDataNCF:

  def __init__(self) -> None:
    pass


  # Ideally:
  """
  We need Only 3 columns User Id, Item Id and Rating or interaction column. 
  - Filter data to remove nan or 0 ratings. 
  - Iterative clean the data set to remove a minimum of X users, our case 5
  - Decide if we want to :
    - Save 3 files: train.ratings, negative.test, test.ratings
    - Save 2 files: train.ratigs, test.ratings 
  For the second approach we have to create a dataset and redo the train NCF file. 
  Caviats,
  For 3 files:
   - We have a script which do not ensure that there is a minimum of X interaction. We are only
     makeing sure that the users with at least 1 interaction are set in the traiing file. 
   - We are using sparse matrix creataion which can be compututationally expensive. 
   - We have to create new evaluation if we want to check MRR. 
   
  """