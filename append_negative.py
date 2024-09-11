import pandas as pd
import numpy as np

# Load the dataset with only positive interactions
data_path = 'data/customer_product_interactions.csv'
data = pd.read_csv(data_path)

# Get all unique customers and products
unique_customers = data['customer_id'].unique()
unique_products = data['product_id'].unique()

# Create a DataFrame with all possible customer-product combinations
all_customer_product_pairs = pd.MultiIndex.from_product([unique_customers, unique_products], names=['customer_id', 'product_id']).to_frame(index=False)

# Merge with the original dataset to find all possible interactions
all_interactions = pd.merge(all_customer_product_pairs, data, on=['customer_id', 'product_id'], how='left')

# Fill missing values in the 'interaction' column with 0 (indicating no purchase)
all_interactions['interaction'].fillna(0, inplace=True)

# Split into positive (interaction > 0) and negative (interaction = 0) samples
positive_interactions = all_interactions[all_interactions['interaction'] > 0]
negative_interactions = all_interactions[all_interactions['interaction'] == 0]

# Randomly sample negative interactions (you can control the size by adjusting frac)
negative_samples = negative_interactions.sample(frac=0.1, random_state=42)  # Sample 10% of negative interactions

# Combine the positive interactions with the sampled negative interactions
final_data = pd.concat([positive_interactions, negative_samples])

# Save the final dataset with both positive and negative samples
final_data.to_csv('data/customer_product_interactions_with_negatives.csv', index=False)

print(f"Original data size: {data.shape[0]}")
print(f"Final data size with negatives: {final_data.shape[0]}")
