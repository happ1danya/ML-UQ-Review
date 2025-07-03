import random
import numpy as np
import tensorflow as tf
import os

# Set random seed for Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  

import pandas as pd

df = pd.read_csv("./datasets/rice_type/riceClassification.csv")

df = df.drop('id', axis=1)
df = df.drop('MajorAxisLength', axis=1)
y = df['Class']
X = df.drop('Class', axis=1)


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

from tensorflow.keras.utils import to_categorical
# Label Onehot-encoding 
y_Onehot = to_categorical(y)

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y_Onehot, random_state=SEED , test_size=0.25)

from sklearn.preprocessing import StandardScaler 
# object from StandardScaler 
scaler = StandardScaler() 
# Scale the data
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

import numpy as np
np.save('./datasets/rice_type/X_test_rt.npy', X_test_scaled)
np.save('./datasets/rice_type/y_test_rt.npy', y_test)

import keras
from keras import layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train_scaled.shape[1]]),
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

model.build(input_shape=(None,X_train_scaled.shape[1]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    batch_size=128,
    epochs=200,
    callbacks=[early_stopping],
)

model.save('./datasets/rice_type/rt.keras')