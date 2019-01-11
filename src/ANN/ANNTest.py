from object_management.object_manager import ObjectManager
import keras
import gensim
import numpy as np
from sklearn.metrics import hamming_loss
import ast
import os

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

modelFiles = os.listdir("D:\\Okul Dosyalar\\Ders\\BBM 406\\ProjeAykutGit\\bbm406-project-what-is-this-books-genre\\src\\ANN\\Model")

word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('../2vecmodels/GoogleNews-vectors-negative300.bin', binary=True)
objManager = ObjectManager()
testData = objManager.ReadObject("D:\\Okul Dosyalar\\Ders\\BBM 406\\Project\\What-is-this-book-s-genre-\\RecordedObject\\TestData")
file = open("log.txt","w")
for modelName in modelFiles:
    file.write(str(modelName) + "\n")
    model = keras.models.load_model(
'D:\\Okul Dosyalar\\Ders\\BBM 406\\ProjeAykutGit\\bbm406-project-what-is-this-books-genre\\src\\ANN\\Model\\' + str(modelName) )

    #  -- Test Data --
    testSummary = testData["Summary"]  # novel's summaries for test
    testGenre = testData["Genre"]  # novel's genres for test

    tempTest = Word2Vec(word2vecModel, testSummary)
    x_test = np.array(tempTest)

    y_test = ConvertGenreListToVector(testData)
    # Multilabel classifier
    trasholdList = [0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40, 0.45, 0.50]
    # -My Prediction-
    for th in trasholdList:
        print("-- Trashold Value: ", th, " --")
        file.write(str(modelName) + "\n")

        predict = model.predict(x_test, 32)
        predict[predict >= th] = 1
        predict[predict < th] = 0
        predictionHit = 0
        for i in range(predict.shape[0]):
            pre = [i for i, e in enumerate(predict[i]) if e == 1]
            acc = [i for i, e in enumerate(y_test[i]) if e == 1]
            # if len(pre) == 0:
            #     break
            hitNum = 0
            for j in pre:
                if acc.__contains__(j):
                    hitNum += 1
            hitRate = hitNum / len(acc)
            if hitRate > 0.7:
                predictionHit += 1
        print('Test accuracy:', (predictionHit / predict.shape[0]))
        file.write('Test accuracy:'+ str(predictionHit / predict.shape[0]) + "\n")
    print("Hamming Loss:", hamming_loss(predict, np.array(y_test)))
    file.write("Hamming Loss:" + str(hamming_loss(predict, np.array(y_test))) + "\n")
file.close()
print("Done")

