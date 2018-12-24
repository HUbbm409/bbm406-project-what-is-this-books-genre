# Importing Required Packages
# ---------------------------

import pandas as pd
import numpy as np
import pickle
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
from data_management.dataset_manager import DatasetManager

# -- Get data --
manager = DatasetManager()
data = manager.ReadCleanedData()  # read data

#  -- Train data --
trainSize = int(len(data) * 0.8)  # train data size %80, validation data size %20
trainSummary = data["Summary"][:trainSize]  # novel's summaries for train
trainGenre = data["Genre"][:trainSize]  # novel's genres for train

#  -- Test Data --
testSummary = data["Summary"][trainSize:]  # novel's summaries for test
testGenre = data["Genre"][trainSize:]  # novel's genres for test

# -- Tokenize and Prepare Vocabulary --
# 20 news groups
num_labels = 15  # class number. We have 22 layer for classification
vocab_size = 15000  # inputs number for neural network, It must be examine in feature for our data set
batch_size = 250

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)  # tokenizer object, number of words ???
tokenizer.fit_on_texts(trainSummary)    # bag of words

x_train = tokenizer.texts_to_matrix(trainSummary, mode='tfidf')  # apply tfidf to train
x_test = tokenizer.texts_to_matrix(testSummary, mode='tfidf')   # apply tfidf to test


"""
At learning time, this simply consists in learning one regressor or binary classifier per class. In doing so, one needs 
to convert multi-class labels to binary labels (belong or does not belong to the class). LabelBinarizer makes this 
process easy with the transform method.
"""
encoder = LabelBinarizer()
encoder.fit(trainGenre)
y_train = encoder.transform(trainGenre)  # convert to train output data to binary
y_test = encoder.transform(testGenre)  # convert to  test output data to binary

# -- Build Keras Model and Fit --
# Now neural network model like that: [15000-256-256-15]
model = Sequential()
model.add(Dense(256, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

# this parameters haven't know what are they used for yet
# it must be examine and would be found appropriates
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=5,
                    verbose=1,
                    validation_split=0.1)

# -- Evaluate model --
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])  # I reached %40 accuracy for now
print("DONE")