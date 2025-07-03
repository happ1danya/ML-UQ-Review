from ucimlrepo import fetch_ucirepo 
import random
import tensorflow as tf
import os
import numpy as np
  
# Set random seed for Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  

# fetch dataset 
magic_gamma_telescope = fetch_ucirepo(id=159) 
  
# data (as pandas dataframes) 
X = magic_gamma_telescope.data.features 
y = magic_gamma_telescope.data.targets 

import numpy as np
from collections import Counter
def analyze_class_imbalance(labels):
    """
    Analyze class imbalance for binary or multiclass classification.
    
    Args:
        labels (list or array-like): The list of class labels.
        
    Returns:
        dict: Contains class counts and max-to-min ratio.
    """
    counts = Counter(labels)
    class_counts = dict(counts)
    
    max_count = max(counts.values())
    min_count = min(counts.values())
    
    max_to_min_ratio = max_count / min_count if min_count > 0 else np.inf
    
    results = {
        'class_counts': class_counts,
        'max_to_min_ratio': max_to_min_ratio
    }
    
    return results

# Convert y DataFrame to a list of labels
labels = y.iloc[:, 0].tolist()

import pandas as pd

y_onehot = pd.get_dummies(y)

from sklearn.model_selection import train_test_split


# Split the data into train and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.25, random_state=SEED)

np.save('./datasets/gamma/X_test_gamma.npy', X_test)
np.save('./datasets/gamma/y_test_gamma.npy', y_test)

import keras
from keras import layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train.shape[1]]),
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(128,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(64,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(32,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(16,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(2,activation='softmax')
])



model.build(input_shape=(None,X_train.shape[1]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=128,
    epochs=200,
    callbacks=[early_stopping],
)

model.save('./datasets/gamma/gamma.keras')