from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from data_management.dataset_manager import DatasetManager
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem import WordNetLemmatizer
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
# nltk.download("wordnet")
# Model as pipeline
model = Pipeline([
    ('vect', CountVectorizer()),
    ('tf', TfidfTransformer(use_idf=False)),
    ('norm', Normalizer()),
    ('clf', MultinomialNB()),
])
# Parameters to be tested
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'vect__stop_words': ['english', None],
              'clf__alpha': [1e-2, 1e-3]
}
# Numerise genre types
genre_types = {" Fiction": 0, " Novel": 1, " Fantasy": 2, " Children's literature": 3, " Mystery": 4,
               " Young adult literature": 5, " Suspense": 6, " Historical novel": 7, " Thriller": 8,
               " Horror": 9, " Romance novel": 10, " Adventure novel": 11, " Non-fiction": 12,
               " Alternate history": 13, " Comedy": 14}

# Read cleaned data
dataset_manager = DatasetManager()
data = dataset_manager.ReadCleanedData()
# Prepare data

train, validation, test = dataset_manager.SplitData()

t_summaries = dataset_manager.Stemmize(train["Summary"])
v_summaries = dataset_manager.Stemmize(validation["Summary"])

t_genres = [genre_types[genre] for genre in train["Genre"]]
v_genres = [genre_types[genre] for genre in validation["Genre"]]
print("Data prepared")

# Train model
gs_clf = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(t_summaries, t_genres)

s_pred = gs_clf.best_estimator_.predict(v_summaries)

print(gs_clf.best_score_, gs_clf.best_params_)
print(confusion_matrix(v_genres, s_pred))

# Plot confusion matrix
df_matrix = pd.DataFrame(confusion_matrix(v_genres, s_pred), index=[key for key in genre_types.keys()],
                         columns=[key for key in genre_types.keys()])
plt.figure(figsize = (10,7))
sn.heatmap(df_matrix, annot=True, fmt='g')
plt.show()
# print(confusion_matrix(validation_genres, predictions))
