from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from data_management.dataset_manager import DatasetManager
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import ast
# Model as pipeline

model = Pipeline([
    ('vect', CountVectorizer()),
    ('tf', TfidfTransformer(use_idf=True)),
    ('norm', Normalizer()),
    ('clf', MultinomialNB([1] * 27)),
])
# Parameters to be tested
parameters = {'vect__ngram_range': [(1, 2)],
              'vect__stop_words': [None],
              'clf__alpha': [1e-3]
}
# Numerise genre types
genre_types = {' Fiction': 0, ' Speculative fiction': 1, ' Science Fiction': 2, ' Novel': 3, ' Fantasy': 4,
               " Children's literature": 5, ' Mystery': 6, ' Young adult literature': 7, ' Suspense': 8,
               ' Crime Fiction': 9, ' Historical novel': 10, ' Thriller': 11, ' Horror': 12, ' Romance novel': 13,
               ' Historical fiction': 14, ' Detective fiction': 15, ' Adventure novel': 16, ' Non-fiction': 17,
               ' Alternate history': 18, ' Spy fiction': 19, ' Comedy': 20, ' Dystopia': 21, ' Autobiography': 22,
               ' Satire': 23, ' Gothic fiction': 24, ' Comic novel': 25, ' Biography': 26}
reverse_genre_types = {v: k for k, v in genre_types.items()}

# Read cleaned data
dataset_manager = DatasetManager()
data = dataset_manager.ReadCleanedData()
# Prepare data

train, test, genre_set = dataset_manager.SplitDataMultipleGenre()
print(len(test))
t_summaries = train["Summary"]
test_summaries = test["Summary"]

t_genres = [genre_types[genre] for genre in train["Genre"]]
test_genres = [genre_types[genre] for genre in test["Genre"]]
test_genre_list = []
for genres in test["GenreList"]:
    genres = ast.literal_eval(genres)
    temp = []
    for genre in genres:
        temp.append(genre_types[genre])
    test_genre_list.append(temp)

print("Data prepared")

# Train model
gs_clf = GridSearchCV(model, parameters, n_jobs=-2)
gs_clf = gs_clf.fit(t_summaries, t_genres)

s_pred = gs_clf.best_estimator_.predict(test_summaries)

print(gs_clf.best_score_, gs_clf.best_params_)
print(np.mean(s_pred == test_genres))
print(confusion_matrix(test_genres, s_pred))

hit = 0
for i in range(len(s_pred)):
    if s_pred[i] in test_genre_list[i]:
        hit += 1
print(hit)

# Plot confusion matrix
df_matrix = pd.DataFrame(confusion_matrix(test_genres, s_pred), index=[key for key in genre_types.keys()],
                         columns=[key for key in genre_types.keys()])
plt.figure(figsize = (10,10))
sn.heatmap(df_matrix, annot=True, fmt='g')
plt.show()
plt.savefig("naive_bayes.png")
# print(confusion_matrix(validation_genres, predictions))
