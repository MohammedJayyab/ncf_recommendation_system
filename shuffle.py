import pandas as pd
import random

# Load CSV file
csv_file = 'data/customer_product_interactions_with_binary.csv'
df = pd.read_csv(csv_file)

# Shuffle the rows of the DataFrame
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame back to a new CSV file
df_shuffled.to_csv('shuffled_file.csv', index=False)

print("The CSV file has been shuffled and saved as 'shuffled_file.csv'.")
