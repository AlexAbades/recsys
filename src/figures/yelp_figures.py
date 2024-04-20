import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.tools.tools import TextLogger

# filename = "/work3/s212784/data/processed/YELP/data_yelp.csv"
filename = "/work3/s212784/data/processed/YELP/yelp.csv"

user_column = "userId"
item_column = "businessId"
interaction_column = "stars"
sep = "\t"
bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, np.inf]
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10-20', '20-30', '>30']

logger = TextLogger("yelp_figures.log")

df = pd.read_csv(filename, usecols=[user_column, item_column, interaction_column], sep=sep)
logger.log(df.head())

# Assuming df is your original DataFrame with user data
user_interactions_sorted = df.groupby(user_column)[item_column].count().sort_values(ascending=False)
sorted_counts_df_user = user_interactions_sorted.reset_index()
sorted_counts_df_user.columns = ['User', 'Count of Items']

logger.log("data transformed")

plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_counts_df_user)), sorted_counts_df_user['Count of Items'])

# Set the y-axis to have logarithmic labels
y_vals = sorted_counts_df_user['Count of Items']
y_ticks = [1, 10, 100, 1000, 10000, 100000, 1000000]  # You can adjust these values based on your data range
plt.yticks(y_ticks, labels=[f'10^{np.log10(y).astype(int)}' if y > 0 else '0' for y in y_ticks])
plt.xlabel('User')
plt.ylabel('Logarithmic Count of Items (like)')
plt.title('Bar Plot with Custom Logarithmic Labels')
plt.xticks(rotation=45)  # Rotate for better label readability
plt.savefig("users_interaction_dist.jpg")







# plt.figure(figsize=(10, 6))
# plt.bar(range(len(sorted_counts_df_user)), sorted_counts_df_user['Count of Items'], log=True)  # Logarithmic y-axis
# plt.xlabel('User')
# plt.ylabel('Count of Items')
# plt.title('Logarithmic Count of Items per User')
# plt.xticks(range(len(sorted_counts_df_user)), sorted_counts_df_user['User'], rotation=45)  # Apply rotation for better label readability
# 









# logger.log("Data loaded")
# logger.log(df.head())

# # Interaction per user
# interaction_counts_user = df.groupby(user_column)[interaction_column].size()
# user_interactions_sorted = interaction_counts_user.sort_values(ascending=False).copy()
# sorted_counts_df_user = user_interactions_sorted.reset_index()
# sorted_counts_df_user.columns = ['User', 'Count of Items'] 

# # Plot Users
# plt.figure(figsize=(12, 8))
# plt.bar(
#     range(len(sorted_counts_df_user)), sorted_counts_df_user["Count of Items"]
# )
# plt.xlabel("User Id")
# plt.ylabel("Nº of interactions")
# plt.title("Count of Interacted Items per User")
# plt.xticks(rotation=45)
# plt.ylim(0, max(sorted_counts_df_user["Count of Items"]) + 100) 
# plt.savefig("users_interaction.jpg")
# logger.log("Interaction per user done")

# # Interaction per item
# interaction_counts_item = df.groupby(item_column)[interaction_column].size()
# item_interactions_sorted = interaction_counts_item.sort_values(ascending=False).copy()
# sorted_counts_df_item = item_interactions_sorted.reset_index()
# sorted_counts_df_item.columns = ['Item', 'Count of Users']

# # Plot Items
# plt.figure(figsize=(12, 8))
# plt.bar(
#     range(len(sorted_counts_df_item)), sorted_counts_df_item["Count of Users"]
# )
# plt.xlabel("Item Id")
# plt.ylabel("Nº of interactions")
# plt.title("Count of Interacted Users per Item")
# plt.xticks(rotation=45)
# plt.ylim(0, max(sorted_counts_df_item["Count of Items"]) + 100) 
# plt.savefig("items_interaction.jpg")

# logger.log("Interaction per uitem done")

# # Bin the data into the specified categories
# interaction_categories_user = pd.cut(interaction_counts_user, bins=bins, labels=labels, right=False)
# interaction_categories_item = pd.cut(interaction_counts_item, bins=bins, labels=labels, right=False)
# logger.log("Binning done")

# # Count the number of users in each category
# category_counts_user = interaction_categories_user.value_counts().sort_index()
# category_counts_item = interaction_categories_item.value_counts().sort_index()
# logger.log("Counting done")

# #  Plot User categories
# plt.figure(figsize=(12, 8))
# plt.bar(category_counts_user.index, category_counts_user.values)
# plt.xlabel('Number of Interactions')
# plt.ylabel('Number of Users')
# plt.title('Number of Users by Interaction Categories')
# plt.xticks(rotation=45) 
# plt.savefig("users_interaction_category.jpg")
# category_counts_user.to_csv("frequency_intercation_user.csv")

# # Plot Item categories 
# plt.figure(figsize=(12, 8))
# plt.bar(category_counts_item.index, category_counts_item.values)
# plt.xlabel('Number of Interactions')
# plt.ylabel('Number of Items')
# plt.title('Number of Items by Interaction Categories')
# plt.xticks(rotation=45) 
# plt.savefig("item_interaction_category.jpg")
# category_counts_item.to_csv("frequency_intercation_item.csv")