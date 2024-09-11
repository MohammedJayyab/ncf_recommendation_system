import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.ncf_model import build_ncf_model  # Import from models directory

# Load the dataset for encoding purposes
data_path = 'data/customer_product_interactions_with_binary.csv'  # Use the path to your CSV file

# Load the data, specifying the dtype to avoid mixed types warnings
data = pd.read_csv(data_path, dtype={'customer_id': str, 'product_id': str})

# Replace NaN descriptions with a placeholder
data['description'].fillna("No description available", inplace=True)

# Before encoding, check if the original customer ID is in the dataset
def customer_exists(customer_id, data):
    return customer_id in data['customer_id'].values

# Function to get a list of products the customer has already purchased
def get_purchased_products(customer_id, data):
    return data[data['customer_id'] == customer_id]['product_id'].values

# Example usage
if __name__ == '__main__':
    customer_id = '13755'  # Specify the customer ID as a string : 17850 , 13047, 13755
    top_n = 5  # Specify how many top products to recommend
    
    # Check if the customer ID exists in the original dataset (before encoding)
    if not customer_exists(customer_id, data):
        print(f"Customer ID {customer_id} not found in the dataset.")
        exit()  # Exit if the customer ID is not found

    # Now encode the customer_id using LabelEncoder
    customer_encoder = LabelEncoder()

    # Apply LabelEncoder to customer_id
    data['customer_id'] = customer_encoder.fit_transform(data['customer_id'])

    num_customers = data['customer_id'].nunique()
    num_items = data['product_id'].nunique()

    # Rebuild the model structure and load trained weights
    ncf_model = build_ncf_model(num_customers, num_items)
    ncf_model.load_weights('models/ncf_model.weights.h5')  # Load the previously saved weights

    # Function to recommend top N products for a given customer
    def recommend_top_n(customer_id, top_n=5):
        # Get the encoded customer ID
        encoded_customer = customer_encoder.transform([customer_id])[0]
        
        # Create item input for all products
        item_input = np.array([i for i in range(num_items)])
        customer_input = np.array([encoded_customer] * num_items)
        
        # Predict interaction scores for all products for the given customer
        predictions = ncf_model.predict([customer_input, item_input])
        
        # Get the products the customer has already purchased
        purchased_products = get_purchased_products(customer_id, data)
        
        # Sort the predictions and get top-N products
        top_n_product_indices = predictions.flatten().argsort()[::-1]  # Sort in descending order
        
        # Filter out already purchased products
        top_n_product_indices = [i for i in top_n_product_indices if data['product_id'].iloc[i] not in purchased_products]
        
        # Limit to top N recommendations
        top_n_product_indices = top_n_product_indices[:top_n]
        
        # Get the recommended product IDs (no need to encode/decode again)
        recommended_products = data['product_id'].iloc[top_n_product_indices].values
        
        # Get descriptions for the top N products using the original product ID, trimming whitespace
        recommendations = [(prod_id, data[data['product_id'] == prod_id]['description'].values[0].strip()) for prod_id in recommended_products]
        
        return recommendations

    # Run the recommendation process
    recommendations = recommend_top_n(customer_id, top_n)
    
    if len(recommendations) > 0:  # Check if the recommendations list is not empty
        print(f"Top {top_n} product recommendations for customer {customer_id}:")
        for product_id, description in recommendations:
            print(f"Product ID: {product_id}, Description: {description}")
    else:
        print(f"No recommendations available for customer {customer_id}.")
