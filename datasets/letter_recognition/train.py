
import numpy as np
import random
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

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
letter_recognition = fetch_ucirepo(id=59) 
  
# data (as pandas dataframes) 
X = letter_recognition.data.features 
y = letter_recognition.data.targets 
  
# metadata 
print(letter_recognition.metadata) 
  
# variable information 
print(letter_recognition.variables) 



# Define column names exactly as in .names (plus the letter label)
# col_names = [
#     'letter', 'x-box', 'y-box', 'width', 'high', 'onpix',
#     'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar',
#     'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'
# ]

# df = pd.read_csv(
#     './datasets/letter_recognition/letter-recognition.data',
#     header=None,            # no header row in the file
#     names=col_names         # assign our list of names
# )

# # Convert pixel (integer) columns to int type and letter to category
# int_cols = col_names[1:]
# df[int_cols] = df[int_cols].astype(int)
# df['letter'] = df['letter'].astype('category')

# y = df['letter']
# X = df.drop(columns=['letter'])


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# if y is a pandas Series of letters:
#  e.g. y = df['letter']

# 1) map letters → 0…25
le = LabelEncoder()
y_int = le.fit_transform(y)       # array of shape (n_samples,), values 0–25

# 2) one-hot encode
y_onehot = to_categorical(y_int)  # shape (n_samples, 26)

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y_onehot, test_size=0.25, random_state=SEED)

np.save('./datasets/letter_recognition/X_test_lr.npy', X_test)
np.save('./datasets/letter_recognition/y_test_lr.npy', y_test)

import keras
from keras import layers

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train.shape[1]]),
    layers.Dense(200,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(100,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(26,activation='softmax')
])

    

model.build(input_shape=(None,X_train.shape[1]))

from keras.optimizers import SGD

callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)
# 定義訓練方式  
sgd = SGD(learning_rate = 0.001, momentum = 0.95)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
  
# 開始訓練  
train_history = model.fit(x=X_train,  
                          y=y_train, validation_split=0.2,  
                          epochs=500, batch_size=64, verbose=1, callbacks=[callback])


model.save('./datasets/letter_recognition/lr.keras')