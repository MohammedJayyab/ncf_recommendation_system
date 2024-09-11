import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.ncf_model import build_ncf_model  # Import from models directory

# Load the dataset for encoding purposes
data_path = 'data/customer_product_interactions_with_binary.csv'  # Use the path to your CSV file
data = pd.read_csv(data_path)

# Before encoding, check if the original customer ID is in the dataset
def customer_exists(customer_id, data):
    if customer_id in data['customer_id'].values:
        return True
    return False

# Function to get a list of products the customer has already purchased
def get_purchased_products(customer_id, data):
    return data[data['customer_id'] == customer_id]['product_id'].values

# Example usage
if __name__ == '__main__':
    customer_id = '17850'  # Specify the customer ID
    top_n = 5  # Specify how many top products to recommend
    
    # Check if the customer ID exists in the original dataset (before encoding)
    if not customer_exists(customer_id, data):
        print(f"Customer ID {customer_id} not found in the dataset.")
        exit()  # Exit if the customer ID is not found

    # Now encode the customer_id and product_id using LabelEncoder
    customer_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    data['customer_id'] = customer_encoder.fit_transform(data['customer_id'])
    data['product_id'] = item_encoder.fit_transform(data['product_id'])  # product_id is now text, no issue for LabelEncoder

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
        encoded_purchased_products = item_encoder.transform(purchased_products)
        
        # Sort the predictions and get top-N products
        top_n_product_indices = predictions.flatten().argsort()[::-1]  # Sort in descending order
        
        # Filter out already purchased products
        top_n_product_indices = [i for i in top_n_product_indices if i not in encoded_purchased_products]
        
        # Limit to top N recommendations
        top_n_product_indices = top_n_product_indices[:top_n]
        
        # Decode product IDs back to original product IDs (which are text)
        top_n_products = item_encoder.inverse_transform(top_n_product_indices)
        
        return top_n_products

    # Run the recommendation process
    recommendations = recommend_top_n(customer_id, top_n)
    
    if len(recommendations) > 0:  # Check if the recommendations list is not empty
        print(f"Top {top_n} product recommendations for customer {customer_id}: {recommendations}")
    else:
        print(f"No recommendations available for customer {customer_id}.")
