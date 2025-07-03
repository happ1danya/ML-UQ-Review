import random
import pandas as pd
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

# load the data from csv file placed locally in our pc
df = pd.read_csv('./datasets/fraud_detection/creditcard.csv')

from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')

# Step 1: Separate features and labels
y = df['Class']
X = df.drop(columns='Class', axis=1)

# Step 2: One-hot encode labels
from tensorflow.keras.utils import to_categorical
y_Onehot = to_categorical(y)

# Step 3: Split
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)

for train_idx, test_idx in splitter.split(X, y):  # split using y (NOT y_Onehot)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_Onehot[train_idx], y_Onehot[test_idx]

import numpy as np
np.save('./datasets/fraud_detection/X_test_fd.npy', X_test)
np.save('./datasets/fraud_detection/y_test_fd.npy', y_test)

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

model.save('./datasets/fraud_detection/fd.keras')