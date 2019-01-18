from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from data_management.dataset_manager import DatasetManager
from object_management.object_manager import ObjectManager
import gensim
import numpy as np
import ast
from pathlib import Path


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


def ConvertGenreListToVector(data):
    genreTypes = {' Fiction': 0, ' Speculative fiction': 1, ' Science Fiction': 2, ' Novel': 3, ' Fantasy': 4,
                  " Children's literature": 5, ' Mystery': 6, ' Young adult literature': 7, ' Suspense': 8,
                  ' Crime Fiction': 9, ' Historical novel': 10, ' Thriller': 11, ' Horror': 12, ' Romance novel': 13,
                  ' Historical fiction': 14, ' Detective fiction': 15, ' Adventure novel': 16, ' Non-fiction': 17,
                  ' Alternate history': 18, ' Spy fiction': 19, ' Comedy': 20, ' Dystopia': 21, ' Autobiography': 22,
                  ' Satire': 23, ' Gothic fiction': 24, ' Comic novel': 25, ' Biography': 26}
    labels = []  # represent books all genres as a vector
    for genreList in data["GenreList"]:
        vector = [0] * 27
        genres = ast.literal_eval(genreList)
        for genre in genres:
            vector[genreTypes.get(genre)] = 1.0
        labels.append(vector)
    return np.array(labels)


def BuildNeuralNetworkModel(HiddenLayerNumber, NeuronNumber, ActivationFunction, InputDimention,
                            OutputActivationFunction, DropOutValue, OutputLayerNumber):
    model = Sequential()
    for i in range(1, HiddenLayerNumber + 1):
        if i == 1:
            model.add(Dense(NeuronNumber, input_dim=InputDimention, activation=ActivationFunction))
            if DropOutValue > 0 and DropOutValue < 1:
                model.add(Dropout(DropOutValue))
        if i == HiddenLayerNumber:
            model.add(Dense(OutputLayerNumber, activation=OutputActivationFunction))
        else:
            model.add(Dense(NeuronNumber, activation=ActivationFunction))
            if DropOutValue > 0 and DropOutValue < 1:
                model.add(Dropout(DropOutValue))
    model.summary()
    return model

def CompileAndFitTheModel(Model, BatchSize, EpochSize, XTrain, YTrain):
    Model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = Model.fit(XTrain, YTrain,
                        batch_size=BatchSize,
                        epochs=EpochSize,
                        verbose=1,
                        validation_split=0.1)
    return Model


# -- DATA PREPARATION--

manager = DatasetManager()
data = manager.ReadCleanedData()  # read data
trainData, testData, uniqueGenreList = manager.SplitDataMultipleGenre(data, 20)
word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('../2vecmodels/GoogleNews-vectors-negative300.bin', binary=True)

# -- Train Data --

trainSummary = trainData["Summary"]  # novel's summaries for train
trainGenre = trainData["Genre"]  # novel's genres for train

tempTrain = Word2Vec(word2vecModel, trainSummary)
x_train = np.array(tempTrain)
y_test = ConvertGenreListToVector(trainData)

# -- CREATE MODEL --
model = BuildNeuralNetworkModel(3, 256, "relu", 300, "sigmoid", 0.5, 27)

epochList = [1, 5, 10, 25, 50, 100, 150, 200, 250,100, 500, 1000]

batchSizeList = [2, 4, 8, 16,64]
# -- TRAIN MODEL --

directory = Path().absolute()
for epoch in epochList:
    for batch in batchSizeList:
        model = CompileAndFitTheModel(model, batch, epoch, x_train,y_test)
        modelName = "\\model" + str(epoch) + "Epoch" + str(batch) + "Batch" + ".h5"
        model.model.save(str(directory) +"\\Model" +modelName)
print("DONE!!")
recorder = ObjectManager()
recorder.RecordObject(testData,str(directory)+"\\RecordedObject\\TestData")
