import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

aggregated_counts_user = pd.DataFrame()
aggregated_counts_item = pd.DataFrame()



filename = "/work3/s212784/data/processed/YELP/data_yelp.csv"
user_column = "userId"
item_column = "businessId"
interaction_column = "stars"

df_iter = pd.read_csv(filename, usecols=[user_column, item_column, interaction_column] , iterator=True, chunksize=100000)

for i, df_chunk in enumerate(df_iter):
    if not i:
        print(df_chunk.memory_usage(index=True))
        print(f"Total: {(df_chunk.memory_usage(index=True).sum())/1000000000}")
    # Aggregate counts per user for the chunk
    chunk_counts_user = (
        df_chunk.groupby(user_column)[interaction_column]
        .count()
        .reset_index(name="interaction_per_user")
    )
    chunk_counts_item = (
        df_chunk.groupby(item_column)[interaction_column]
        .count()
        .reset_index(name="interaction_per_item")
    )
    # Append to the aggregated DataFrame
    aggregated_counts_user = pd.concat([aggregated_counts_user, chunk_counts_user])
    aggregated_counts_item = pd.concat([aggregated_counts_item, chunk_counts_item])

# Now, aggregate counts across all chunks (summing up counts if a user appears in multiple chunks)
final_counts_user = aggregated_counts_user.groupby(user_column).sum().reset_index()
final_counts_item = aggregated_counts_item.groupby(item_column).sum().reset_index()

# Sort values
sorted_counts_df_user = final_counts_user.sort_values(
    by="interaction_per_user", ascending=False
)
sorted_counts_df_item = final_counts_item.sort_values(
    by="interaction_per_item", ascending=False
)

# Sort values
sorted_counts_df_user.columns = ["User", "interaction_per_user"]
sorted_counts_df_item.columns = ["Item", "interaction_per_item"]

# Binning
bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, np.inf]
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10-20", "20-30", ">30"]

# Apply categorization
final_counts_user["interaction_category"] = pd.cut(
    final_counts_user["interaction_per_user"], bins=bins, labels=labels, right=False
)
final_counts_item["interaction_category"] = pd.cut(
    final_counts_item["interaction_per_item"], bins=bins, labels=labels, right=False
)

# Aggregate category counts
category_counts_user = (
    final_counts_user["interaction_category"].value_counts().sort_index()
)
category_counts_item = (
    final_counts_item["interaction_category"].value_counts().sort_index()
)


# Plot Users
plt.figure(figsize=(12, 8))
plt.bar(
    range(len(sorted_counts_df_user)), sorted_counts_df_user["interaction_per_user"]
)
plt.xlabel("User")
plt.ylabel("Count of Items")
plt.title("Count of Items per User")
plt.xticks(rotation=45)
plt.savefig("users_interaction.jpg")

# Plot Items
plt.figure(figsize=(12, 8))
plt.bar(
    range(len(sorted_counts_df_item)), sorted_counts_df_item["interaction_per_item"]
)
plt.xlabel("User")
plt.ylabel("Count of Items")
plt.title("Count of Items per User")
plt.xticks(rotation=45)
plt.savefig("items_interaction.jpg")

# Plt Category Interaction User
plt.figure(figsize=(12, 8))
plt.bar(category_counts_user.index.astype(str), category_counts_user.values)
plt.xlabel("Number of Interactions")
plt.ylabel("Number of Users")
plt.title("Number of Users by Interaction Categories")
plt.xticks(rotation=45)
plt.savefig("category_interaction_users.jpg")
category_counts_user_df = category_counts_user.reset_index().rename(columns={'index': 'Interaction Category', 'interaction_category': 'Number of Users'})
category_counts_user_df.to_csv("category_interaction_users.csv", index=False)

# Plt Category Interaction Item
plt.figure(figsize=(12, 8))
plt.bar(category_counts_item.index.astype(str), category_counts_item.values)
plt.xlabel("Number of Interactions")
plt.ylabel("Number of Items")
plt.title("Number of Items by Interaction Categories")
plt.xticks(rotation=45)
plt.savefig("category_interaction_items.jpg")

category_counts_item_df = category_counts_item.reset_index().rename(columns={'index': 'Interaction Category', 'interaction_category': 'Number of Items'})
category_counts_item_df.to_csv("category_interaction_users.csv", index=False)