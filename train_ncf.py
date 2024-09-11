import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight  # Import for class weighting
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error  # Import metrics
from models.ncf_model import build_ncf_model  # Import your NCF model

# Load the dataset with binary interactions
data_path = 'data/customer_product_interactions_with_binary.csv'
data = pd.read_csv(data_path)

# Encode customer_id and product_id using LabelEncoder
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

data['customer_id'] = user_encoder.fit_transform(data['customer_id'])
data['product_id'] = item_encoder.fit_transform(data['product_id'])

num_users = data['customer_id'].nunique()
num_items = data['product_id'].nunique()

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create model inputs (customers, products, and binary interactions)
train_user = train_data['customer_id'].values
train_item = train_data['product_id'].values
train_interaction = train_data['interaction_binary'].values

test_user = test_data['customer_id'].values
test_item = test_data['product_id'].values
test_interaction = test_data['interaction_binary'].values

# Build the NCF model for binary classification
ncf_model = build_ncf_model(num_users, num_items)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_interaction),  # Binary classes (0 and 1)
    y=train_interaction  # Actual interaction labels from the training data
)

# Convert class weights to a dictionary
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Train the model and pass class weights
ncf_model.fit([train_user, train_item], train_interaction, 
              epochs=10, batch_size=64, validation_split=0.1, 
              class_weight=class_weight_dict)  # Pass class weights here

# Save the model weights after training
ncf_model.save_weights('models/ncf_model.weights.h5')

# Predict on the test set (for later evaluation)
predictions = ncf_model.predict([test_user, test_item])
predictions_binary = (predictions > 0.5).astype(int)  # Convert probabilities to binary output (0 or 1)

# Calculate and print the evaluation metrics (same as before)
accuracy = accuracy_score(test_interaction, predictions_binary)
precision = precision_score(test_interaction, predictions_binary, zero_division=0)
recall = recall_score(test_interaction, predictions_binary)
f1 = f1_score(test_interaction, predictions_binary)
mae = mean_absolute_error(test_interaction, predictions)
rmse = np.sqrt(mean_squared_error(test_interaction, predictions))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
