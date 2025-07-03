import random
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import keras


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TextVectorization, Embedding
from keras.models import Sequential  
from keras.layers import Dense,Dropout,LSTM,BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# Set random seed for Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  


df = pd.read_csv("./datasets/tweets/clean_tweet_Dec19ToDec20.csv")

df = df.dropna(axis=0)

X = df['text']
y = df['sentiment']

y_Onehot =  to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y_Onehot, test_size=0.25,random_state = SEED)


max_vocab_length = 50000 # how many words our dictionary will include
max_length = 25 # how many words from a tweet will be included

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode='int',
                                    output_sequence_length=max_length,
                                    standardize='lower_and_strip_punctuation'
                                   )

# fit the text vectorizer to teh train data
text_vectorizer.adapt(X_train)

X_train = text_vectorizer(X_train)
X_test = text_vectorizer(X_test)

np.save('./datasets/tweets/X_test_tweets.npy', X_test)
np.save('./datasets/tweets/y_test_tweets.npy', y_test)


model = Sequential()

model.add(Embedding(max_vocab_length,128))

model.add(LSTM(units=64,dropout=0.2,recurrent_dropout=0.2,kernel_regularizer=keras.regularizers.l1(0.001),seed = SEED))
model.add(Dropout(0.5,seed = SEED))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5,seed = SEED))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5,seed = SEED))
model.add(BatchNormalization())
model.add(Dense(2,activation='softmax'))


callback = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
  
# Train the model
train_history = model.fit(x=X_train, y=y_train, validation_split=0.2,callbacks=[callback], epochs=3, batch_size=128, verbose=1)

# score = model.evaluate(X_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

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

model.save('./datasets/tweets/tweets.keras')