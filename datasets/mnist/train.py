import numpy as np
import random
import tensorflow as tf
import os
import keras


from keras.datasets import mnist  
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D 



# Set random seed for Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  


# Read MNIST data  
(X_train_M, y_train_M), (X_test_M, y_test_M) = mnist.load_data()  

X = np.concatenate((X_train_M, X_test_M))
y = np.concatenate((y_train_M, y_test_M))



# Translation of data  
X_4D = X.reshape(X.shape[0], 28, 28, 1).astype('float32')  

# Standardize
X_4D_norm = X_4D/255

# Label Onehot-encoding 
y_Onehot = to_categorical(y)


X_train, X_test,y_train, y_test = train_test_split(X_4D_norm, y_Onehot, test_size=0.25, random_state=SEED)
np.save('./datasets/mnist/X_test_mnist.npy', X_test)
np.save('./datasets/mnist/y_test_mnist.npy', y_test)


model = Sequential()  

model.add(Conv2D(filters=16,  
                 kernel_size=(3,3),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu',
                 kernel_regularizer =keras.regularizers.l1(0.01),
                 name='conv2d_1')) 
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_1'))  
model.add(Dropout(0.5, name='dropout_1', seed = SEED))
model.add(Conv2D(filters=36,  
                 kernel_size=(3,3),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu',
                 name='conv2d_2'))  
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_2'))  
model.add(Dropout(0.5, name='dropout_2', seed = SEED))
model.add(Flatten(name='flatten_1'))
model.add(Dense(128, activation='relu', name='dense_1'))  
model.add(Dropout(0.5, name='dropout_3', seed = SEED))
model.add(Dense(10, activation='softmax', name='dense_2'))

callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

train_history = model.fit(x=X_train,  
                          y=y_train, validation_split=0.2,  
                          epochs=100, batch_size=256, verbose=1, callbacks=[callback])

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('./datasets/mnist/mnist.keras')