import tensorflow as tf
from tensorflow.keras import layers, Model

def build_ncf_model(num_users, num_items, latent_dim=32):
    # Input layers
    user_input = layers.Input(shape=(1,), name='user_input')
    item_input = layers.Input(shape=(1,), name='item_input')
    
    # Embedding layers (no L2 regularization)
    user_embedding = layers.Embedding(num_users, latent_dim, name='user_embedding')(user_input)
    item_embedding = layers.Embedding(num_items, latent_dim, name='item_embedding')(item_input)
    
    # Flatten the embeddings
    user_embedding = layers.Flatten()(user_embedding)
    item_embedding = layers.Flatten()(item_embedding)
    
    # Concatenate user and item embeddings
    concat_layer = layers.Concatenate()([user_embedding, item_embedding])
    
    # Add dropout to avoid overfitting
    dropout_layer = layers.Dropout(0.3)(concat_layer)
    
    # Fully connected layers without L2 regularization
    dense_1 = layers.Dense(64, activation='relu')(dropout_layer)
    dense_2 = layers.Dense(32, activation='relu')(dense_1)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(dense_2)  # Output layer for binary interaction
    
    # Build and compile the model
    model = Model([user_input, item_input], output)
    
    # Use binary crossentropy as the loss and classification metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy',  # Binary classification loss
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(name='precision'), 
                           tf.keras.metrics.Recall(name='recall')])  # Classification metrics
    
    return model
