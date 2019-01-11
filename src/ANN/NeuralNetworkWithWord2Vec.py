# Importing Required Packages
# ---------------------------

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from data_management.dataset_manager import DatasetManager
import ast
import gensim
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def Word2Vec(model, data):
    temp = []
    for x in data:
        count = 0
        vector = np.zeros((300,))
        for word in x.split(' '):
            try:
                vector += model.get_vector(word)
                count += 1
            except KeyError:
                continue

        if count == 0:
            vector = np.zeros((300,))
            count = 1
            print(x)
        temp.append(vector / count)
    return temp
# -- Get data --


word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('./2vecmodels/GoogleNews-vectors-negative300.bin', binary=True)

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
batch_size = 20
epoch_size = 500

tempTrain = Word2Vec(word2vecModel, trainSummary)
x_train = np.array(tempTrain)

tempTest = Word2Vec(word2vecModel, testSummary)
x_test = np.array(tempTest)

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
model.add(Dense(128, input_shape=(300,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dropout(0.3))
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
s_pred = []
for i in range(predict.shape[0]):
    predictionLabel = predict[i].argmax()
    s_pred.append(predictionLabel)
    label = mostFrequentGenres.get(predictionLabel)
    genreList = testData.iloc[i]["GenreList"]
    genreList = ast.literal_eval(genreList)
    if label in genreList:
        predictionHit += 1


print('Test accuracy:', (predictionHit/predict.shape[0]))

print("Done")
reverse_genre_types = {v: k for k, v in mostFrequentGenres.items()}
test_genres = [reverse_genre_types[genre] for genre in testData["Genre"]]

df_matrix = pd.DataFrame(confusion_matrix(test_genres, s_pred), index=[key for key in reverse_genre_types.keys()],
                         columns=[key for key in reverse_genre_types.keys()])
plt.figure(figsize = (10,10))
sn.heatmap(df_matrix, annot=True, fmt='g')
plt.show()
plt.savefig("naive_bayes.png")