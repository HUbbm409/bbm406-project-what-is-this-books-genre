from object_management.object_manager import ObjectManager
import keras
import gensim
import numpy as np
from sklearn.metrics import hamming_loss
import ast
import os
from pathlib import Path
file = open("Output.txt","w")
genreTypes = {' Fiction': 0, ' Speculative fiction': 1, ' Science Fiction': 2, ' Novel': 3, ' Fantasy': 4,
              " Children's literature": 5, ' Mystery': 6, ' Young adult literature': 7, ' Suspense': 8,
              ' Crime Fiction': 9, ' Historical novel': 10, ' Thriller': 11, ' Horror': 12, ' Romance novel': 13,
              ' Historical fiction': 14, ' Detective fiction': 15, ' Adventure novel': 16, ' Non-fiction': 17,
              ' Alternate history': 18, ' Spy fiction': 19, ' Comedy': 20, ' Dystopia': 21, ' Autobiography': 22,
              ' Satire': 23, ' Gothic fiction': 24, ' Comic novel': 25, ' Biography': 26}
reverse_genre_types = {v: k for k, v in genreTypes.items()}
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

directory = Path().absolute()

word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('../2vecmodels/GoogleNews-vectors-negative300.bin', binary=True)
objManager = ObjectManager()

testData = objManager.ReadObject(str(directory)+"\\RecordedObject\\TestData")
model = keras.models.load_model(str(directory)+"\\Model\\model1000Epoch64Batch.h5")

testSummary = testData["Summary"]  # novel's summaries for test
testGenre = testData["Genre"]  # novel's genres for test

tempTest = Word2Vec(word2vecModel, testSummary)
x_test = np.array(tempTest)


predict = model.predict(x_test, 32)
for i in range(predict.shape[0]):
    bookName = testData.iloc[i]["Title"]
    file.write("---- Book Title: " + str(bookName)+ " ---- \n")
    file.write("---- Actual Genres ---- \n")
    genres = ast.literal_eval(testData.iloc[i]["GenreList"])
    for g in genres:
        file.write(str(g) + " ")
    file.write("\n")
    file.write("\n")

    file.write("---- Predicted Genres ---- \n")
    print(testData.iloc[i]["Title"])
    # print(len(predict[i]))
    for j in range(len(predict[i])):
        if predict[i][j] > 0.1 :
            print(reverse_genre_types.get(j), " " , predict[i][j])
            file.write(str(reverse_genre_types.get(j))+ " " + str(predict[i][j]) + " ")
    file.write("\n")
    file.write("\n")
file.close()


print("Done")