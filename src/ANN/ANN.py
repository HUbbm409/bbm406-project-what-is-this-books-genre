# Importing Required Packages
# ---------------------------

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from data_management.dataset_manager import DatasetManager
from object_management.object_manager import  ObjectManager
import pickle


# -- Get data --
manager = DatasetManager()
data = manager.ReadCleanedData()  # read data
trainData, testData, uniqueGenreList = manager.SplitDataMultipleGenre(data, 20)

#  -- Train data --
trainSummary = trainData["Summary"]  # novel's summaries for train
# trainSummary = manager.Stemmize(trainSummary)  # stemming for train data
# trainSummary = manager.Lemmatize(trainSummary)  # Lemmatize for train data
trainGenre = trainData["Genre"]  # novel's genres for train


# -- Tokenize and Prepare Vocabulary --
# 27 book's groups
num_labels = 27  # class number. We have 22 layer for classification
vocab_size = 20000  # inputs number for neural network, It must be examine in feature for our data set
batch_size = 32
epoch_size = 2

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)  # tokenizer object, number of words ???
tokenizer.fit_on_texts(trainSummary)    # bag of words


x_train = tokenizer.texts_to_matrix(trainSummary, mode='tfidf')  # apply tfidf to train


"""
At learning time, this simply consists in learning one regressor or binary classifier per class. In doing so, one needs 
to convert multi-class labels to binary labels (belong or does not belong to the class). LabelBinarizer makes this 
process easy with the transform method.
"""
encoder = LabelBinarizer()
encoder.fit(trainGenre)
y_train = encoder.transform(trainGenre)  # convert to train output data to binary

# -- Build Keras Model and Fit --
# Now neural network model like that: [15000-256-256-15]

model = Sequential()
model.add(Dense(256, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256, input_shape=(vocab_size,)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(num_labels))

model.add(Activation('sigmoid'))  # original: softmax, multi label: sigmoid
model.summary()

# this parameters haven't know what are they used for yet
# it must be examine and would be found appropriates
# optimizer  = "sgd" = 0,41
model.compile(loss='binary_crossentropy',  # loss = categorical_crossentropy multi label: binary_crossentropy
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch_size,
                    verbose=1,
                    validation_split=0.1)
recorder = ObjectManager()
recorder.RecordObject(testData,"D:\\Okul Dosyalar\\Ders\\BBM 406\\Project\\What-is-this-book-s-genre-\\RecordedObject\\TestData")
# creates a HDF5 file 'my_model.h5'
model.model.save('D:\\Okul Dosyalar\\Ders\\BBM 406\\Project\\What-is-this-book-s-genre-\\RecordedObject\\my_model.h5')
recorder.RecordObject(tokenizer,"D:\\Okul Dosyalar\\Ders\\BBM 406\\Project\\What-is-this-book-s-genre-\\RecordedObject\\Tokenizer")
recorder.RecordObject(encoder,"D:\\Okul Dosyalar\\Ders\\BBM 406\\Project\\What-is-this-book-s-genre-\\RecordedObject\\Encoder")