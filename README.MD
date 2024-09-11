# Neural Collaborative Filtering (NCF) Recommendation System

The Neural Collaborative Filtering (NCF) model is a deep learning-based recommendation system that leverages neural networks to capture complex user-item interaction patterns. In this project, we build an NCF model to recommend products to customers based on their past interactions, using a binary classification approach where interactions are either 1 (purchased) or 0 (not purchased).

## Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Steps to Run](#steps-to-run)
5. [Modeling and Predictions](#modeling)
6. [Evaluation Metrics](#evaluation)
7. [Results](#results)
8. [Future Improvements](#improvements)

## 1. Project Overview <a name="overview"></a>

The goal of this project is to implement a recommendation system using **Neural Collaborative Filtering (NCF)**. The NCF model learns user-item interactions by mapping users and items into dense latent feature vectors and then applies neural networks to predict the probability of interaction between users and items.

The dataset used includes customer transactions from an e-commerce platform, with the objective of predicting future purchases and providing personalized product recommendations.

## 2. Installation <a name="installation"></a>

To set up the environment for this project, follow these steps:

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Required Packages

Install the necessary Python libraries by running:

```bash
pip install -r requirements.txt
```

The required packages include:

- pandas
- numpy
- scikit-learn
- tensorflow

## 3. Data Preparation <a name="data-preparation"></a>

### Step 1: Prepare Binary Interaction Data

Run the `append_interaction_binary.py` script to create a binary interaction dataset:

```bash
python append_interaction_binary.py
```

This script processes the raw transaction data by converting it into a binary interaction matrix. Customers who purchased a product have an interaction value of 1, while those who didn’t have an interaction value of 0.

### Step 2: Sampling Negative Interactions

The script also randomly samples a portion of the negative interactions (interactions where no purchase was made) to balance the dataset and reduce training time.

## 4. Steps to Run <a name="steps-to-run"></a>

### Step 1: Train the NCF Model

Run the `train_ncf.py` script to train the Neural Collaborative Filtering model:

```bash
python train_ncf.py
```

This script trains the NCF model on the prepared dataset, using customer-product interactions as input. The trained model weights are saved to the `models/` directory.

### Step 2: Make Predictions

After training, you can run the `predict.py` script to generate product recommendations for specific customers:

```bash
python predict.py
```

This script loads the trained model and recommends the top-N products for a specified customer.

### Step 3: Evaluate the Model

Run the `train_ncf.py` script to view evaluation metrics such as accuracy, precision, recall, and F1-score:

```bash
python train_ncf.py
```

## 5. Modeling and Predictions <a name="modeling"></a>

The **Neural Collaborative Filtering (NCF)** model used in this project consists of the following components:

- **Embedding Layers**: Maps users and items to dense feature vectors (latent space).
- **Concatenation Layer**: Combines the user and item embeddings.
- **Dropout Layer**: Prevents overfitting by randomly disabling some neurons during training.
- **Dense Layers**: Applies fully connected layers with ReLU activation to learn non-linear interactions between users and items.
- **Output Layer**: A sigmoid function is used to output the probability of interaction between the user and the item.

The model is trained using **binary crossentropy loss** since the task is binary classification (interaction or no interaction).

### Example Prediction

For a specific customer, we generate predictions by feeding all available products through the model and selecting the products with the highest predicted interaction probabilities.

## 6. Evaluation Metrics <a name="evaluation"></a>

We evaluate the performance of the recommendation system using the following metrics:

- **Accuracy**: The proportion of correct predictions (both positive and negative).
- **Precision**: The proportion of predicted positive interactions that are actually correct.
- **Recall**: The proportion of actual positive interactions that were correctly predicted.
- **F1-Score**: The harmonic mean of Precision and Recall.
- **Mean Absolute Error (MAE)**: The average difference between predicted probabilities and actual labels.
- **Root Mean Squared Error (RMSE)**: The square root of the average squared differences between predicted probabilities and actual labels.

### Example Results:

- **Accuracy**: 0.85
- **Precision**: 0.72
- **Recall**: 0.68
- **F1-Score**: 0.70
- **MAE**: 0.12
- **RMSE**: 0.34

## 7. Results <a name="results"></a>

The NCF model recommends products to customers based on their purchase history. For instance, if we run the model for **Customer 13755**, the top recommended products could include:

- Product A: "Stylish Lamp"
- Product B: "Modern Chair"

These recommendations are personalized based on learned interactions and product preferences.

## 8. Future Improvements <a name="improvements"></a>

Some potential areas for improvement in the recommendation system include:

1. **Hyperparameter Tuning**: Experiment with different model architectures, learning rates, and embedding sizes to optimize performance.
2. **Handling Implicit Feedback**: Incorporate implicit feedback techniques, such as weighting interactions based on frequency or recency.
3. **Hybrid Models**: Combine collaborative filtering with content-based filtering to improve recommendations.
4. **Cold Start Problem**: Address the cold start problem for new users or products by integrating additional information, such as product categories or user demographics.

---

This project demonstrates the use of Neural Collaborative Filtering for building a recommendation system using binary customer-product interaction data. The model can be further improved through parameter tuning and hybrid approaches.

## Model Performance

The recommendation model has been evaluated using several key performance metrics. Below is a detailed explanation of each metric and the model's performance:

### 1. **Accuracy: 0.8322 (83.22%)**

- **What it means**: This means that 83.22% of the predictions made by the model are correct. In a binary classification problem like this (predicting interaction vs. no interaction), an accuracy above 80% is typically considered a good performance. However, accuracy alone doesn't give the full picture, especially if the dataset is imbalanced.

### 2. **Precision: 0.7894 (78.94%)**

- **What it means**: Of all the products that the model predicted the user would interact with, 78.94% were correct. Precision is important when you want to minimize false positives (i.e., recommending a product that the user is unlikely to interact with).
- **When to focus on precision**: If your goal is to recommend fewer but more relevant products, increasing precision is essential.

### 3. **Recall: 0.8656 (86.56%)**

- **What it means**: Of all the products that the user actually interacted with, the model correctly predicted 86.56% of them. Recall is important when you want to ensure you recommend all potentially relevant products, even if some irrelevant ones are also recommended.
- **When to focus on recall**: If you want to ensure that users don't miss out on good product recommendations, recall is a critical metric to optimize.

### 4. **F1-Score: 0.8258 (82.58%)**

- **What it means**: The F1-score is the harmonic mean of precision and recall, balancing the two. A score of 82.58% shows that the model has a good balance between making relevant recommendations and minimizing irrelevant ones.
- **Why F1 is useful**: It provides a single score that balances precision and recall, which is useful when you need to balance both avoiding false positives and catching all relevant items.

### 5. **Mean Absolute Error (MAE): 0.2398**

- **What it means**: On average, the model's predicted probability of interaction differs from the actual value by 0.2398. This measures how close the predictions are to the true interactions, with lower values indicating better performance.
- **Why MAE matters**: MAE is a simple and intuitive measure of error, where smaller values mean the model's predictions are closer to the true values.

### 6. **Root Mean Squared Error (RMSE): 0.3434**

- **What it means**: RMSE measures the standard deviation of the prediction errors, penalizing larger errors more heavily than MAE. The value of 0.3434 indicates that, while the model generally makes good predictions, there are some larger errors occasionally.
- **Why RMSE matters**: RMSE is useful when you want to penalize large prediction errors more heavily than small ones.

---

### Summary

- **High recall** (86.56%) means the model is effective at recommending products that users are likely to interact with, ensuring that most relevant items are not missed.
- **Strong precision** (78.94%) ensures that the model doesn’t recommend too many irrelevant products.
- **Accuracy** is solid at 83.22%, but it's important to remember that accuracy alone might not tell the full story if the dataset is imbalanced.
- **Low MAE and RMSE** indicate that the model is making predictions that are close to the actual interactions, with few large errors.

If you want to further improve the model, consider whether you want to prioritize **Precision** (to make fewer but more accurate recommendations) or **Recall** (to ensure you recommend all potentially relevant items).
