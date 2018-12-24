# Importing Required Packages
# ---------------------------

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from data_management.dataset_manager import DatasetManager

# -- Get data --
manager = DatasetManager()
data = manager.ReadCleanedData()  # read data

trainData, testData, uniqueGenreList = manager.SplitDataMultipleGenre(data, 20)
# print(len(data))
#  -- Train data --
# trainSize = int(len(data) * 0.8)  # train data size %80, validation data size %20
trainSummary = trainData["Summary"]  # novel's summaries for train
trainGenre = trainData["Genre"]  # novel's genres for train

#  -- Test Data --
testSummary = testData["Summary"]  # novel's summaries for test
testGenre = testData["Genre"]  # novel's genres for test

# -- Tokenize and Prepare Vocabulary --
# 15 book's groups
num_labels = 27  # class number. We have 22 layer for classification
vocab_size = 20000  # inputs number for neural network, It must be examine in feature for our data set
batch_size = 1
epoch_size = 4

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
model.add(Dense(128, input_shape=(vocab_size,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(128, input_shape=(vocab_size,)))
model.add(Activation('sigmoid'))
model.add(Dense(num_labels))

model.add(Activation('softmax'))  # original: softmax, multi label: sigmoid
model.summary()

# this parameters haven't know what are they used for yet
# it must be examine and would be found appropriates
# optimizer  = "sgd" = 0,41
model.compile(loss='categorical_crossentropy',  # loss = categorical_crossentropy multi label: binary_crossentropy
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch_size,
                    verbose=1,
                    validation_split=0.1)

# -- Evaluate model --
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])
x = model.predict(x_test, batch_size)
# x[x >= 0.1] = 1
# x[x < 0.1] = 0
mostFrequentGenres = {1: ' Alternate history', 2: ' Autobiography', 3: ' Biography', 4: " Children's literature", 5: ' Comedy',
                      6: ' Comic novel', 7: ' Crime Fiction', 8: ' Detective fiction', 9: ' Dystopia', 10: ' Fantasy',
                      11: ' Fiction', 12: ' Gothic fiction', 13: ' Historical fiction', 14: ' Historical novel', 15: ' Horror',
                      16: ' Mystery', 17: ' Non-fiction', 18: ' Novel', 19: ' Romance novel', 20: ' Satire', 21: ' Science Fiction',
                      22: ' Speculative fiction', 23: ' Spy fiction', 24: ' Suspense', 25: ' Thriller', 26: ' Young adult literature',
                      0: ' Adventure novel' }


# -My Prediction-
predict = model.predict(x_test, batch_size)
predictionHit = 0
for i in range(predict.shape[0]):
    predictionLabel = predict[i].argmax()
    label = mostFrequentGenres.get(predictionLabel)
    genreList = testData.iloc[i]["GenreList"]
    if genreList.__contains__(label):
        predictionHit += 1

print('Test accuracy:', (predictionHit/predict.shape[0]))

# -- Hyperparameters--
# 1 hidden layer, drop-out = 0.3 neuron number 512, batch 32,

print("Done")