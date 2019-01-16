import gensim
import numpy as np
from scipy import sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


class InputEnvoy:

    def __init__(self):
        self.word2vec_model = None

    def X_word2vec(self, x_data):
        if self.word2vec_model is None:
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
                 './2vecmodels/GoogleNews-vectors-negative300.bin',
                 binary=True)

        temp = []
        for x in x_data:
            count = 0
            vector = np.zeros((300,))
            not_found_vector = np.full((300,), 0.1)
            for word in x.split(' '):
                try:
                    vector += self.word2vec_model.get_vector(word)
                    count += 1
                except KeyError:
                    vector += not_found_vector
                    count += 1

            if count == 0:
                vector = not_found_vector
                count = 1
                print(x)
            temp.append(vector / count)

        x_data = np.array(temp)
        x_data = sps.lil_matrix(x_data)

        print("Word2Vec model loaded!")
        return x_data

    def Bow(self, corpus, use_idf=True, normalize=False, stop_words='english', ngram_range=(1, 2)):

        vect = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range, use_idf=use_idf)
        vectorized_data = vect.fit_transform(corpus)

        if normalize:
            norm = Normalizer()
            norm.transform(vectorized_data)

        return vectorized_data

    def Extend_features_bow(self, static, extended):
        number_of_features = static.shape[1]
        temp = sps.csr_matrix((extended.data, extended.indices, extended.indptr),
                              shape=(extended.shape[0], number_of_features), copy=True)
        return temp

    def Narrow_features_bow(self):
        pass
