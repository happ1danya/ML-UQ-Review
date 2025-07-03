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

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X = covertype.data.features 
y = covertype.data.targets 


from tensorflow.keras.utils import to_categorical

# shift labels to start from 0
y_shifted = y - 1
y_onehot = to_categorical(y_shifted)

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y_onehot, test_size=0.25, random_state=SEED)

np.save('./datasets/cover_type/X_test_ct.npy', X_test)
np.save('./datasets/cover_type/y_test_ct.npy', y_test)

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
    layers.Dense(7,activation='softmax')
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
                          epochs=500, batch_size=256, verbose=1, callbacks=[callback])

model.save('./datasets/cover_type/ct.keras')