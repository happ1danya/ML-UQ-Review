import pandas as pd
import numpy as np
import keras
import random
import tensorflow as tf
import os

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras import layers
from sklearn.metrics import confusion_matrix


# Set random seed for Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  


df = pd.read_csv('./datasets/neo/nearest-earth-objects(1910-2024).csv')

df = df.drop(['neo_id', 'name', 'orbiting_body'], axis=1)


# Select the columns for imputation
columns_for_imputation = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max']

# Initialize the KNNImputer
imputer = KNNImputer(n_neighbors=5)

# Fit and transform the data
df[columns_for_imputation] = imputer.fit_transform(df[columns_for_imputation])


df['average_diameter'] = (df['estimated_diameter_min'] + df['estimated_diameter_max']) / 2

df['diameter_range'] = df['estimated_diameter_max'] - df['estimated_diameter_min']

df['scaled_relative_velocity'] = (df['relative_velocity'] - df['relative_velocity'].min()) / (df['relative_velocity'].max() - df['relative_velocity'].min())

df['log_miss_distance'] = np.log(df['miss_distance'])

df['velocity_diameter_interaction'] = df['relative_velocity'] * df['average_diameter']

df['velocity_distance_ratio'] = df['relative_velocity'] / df['miss_distance']

df['diameter_magnitude_ratio'] = df['average_diameter'] / df['absolute_magnitude']

# Select numerical columns for normalization
numerical_cols = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max','relative_velocity','miss_distance']

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Assuming 'is_hazardous' is the target variable
X = df.drop('is_hazardous', axis=1)
y = df['is_hazardous']


# Label Onehot-encoding 
y_Onehot = to_categorical(y)

# Split the data into train and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_Onehot, test_size=0.25, random_state=SEED)

np.save('./datasets/neo/X_test_neo.npy', X_test)
np.save('./datasets/neo/y_test_neo.npy', y_test)


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
# # Evaluate the model on the test set
# accuracy = model.evaluate(X_test, y_test)
# print('Accuracy: {}'.format(accuracy))

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Convert predictions and true labels from one-hot encoding to class indices
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_test_classes = np.argmax(y_test, axis=1)

# # Compute the confusion matrix
# conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# # Print the confusion matrix
# print("Confusion Matrix:")
# print(conf_matrix)

model.save('./datasets/neo/neo.keras')
