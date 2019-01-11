from data_management.dataset_manager import DatasetManager
import sklearn.metrics as metrics
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from data_management.input_manager import InputEnvoy

genre_types = {' Fiction': 0, ' Speculative fiction': 1, ' Science Fiction': 2, ' Novel': 3, ' Fantasy': 4,
               " Children's literature": 5, ' Mystery': 6, ' Young adult literature': 7, ' Suspense': 8,
               ' Crime Fiction': 9, ' Historical novel': 10, ' Thriller': 11, ' Horror': 12, ' Romance novel': 13,
               ' Historical fiction': 14, ' Detective fiction': 15, ' Adventure novel': 16, ' Non-fiction': 17,
               ' Alternate history': 18, ' Spy fiction': 19, ' Comedy': 20, ' Dystopia': 21, ' Autobiography': 22,
               ' Satire': 23, ' Gothic fiction': 24, ' Comic novel': 25, ' Biography': 26}
reverse_genre_types = {v: k for k, v in genre_types.items()}
# Setup vectorizer
print(list(genre_types.values()))

# Read cleaned data
dataset_manager = DatasetManager()
input_envoy = InputEnvoy()
data = dataset_manager.ReadCleanedData()
# Prepare data

train, test, genre_set = dataset_manager.SplitDataMultipleGenre()

# Prepare train data
x_train = input_envoy.Bow(train["Summary"], stop_words=None)

y_train = dataset_manager.MultiLabel2Matrix(train, genre_types)
print("Train Data Ready!")

train = None

classifier = LabelPowerset(
    classifier=RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=3),
    require_dense=[False, True]
)

classifier.fit(x_train, y_train)
print("Training Complete!")
# Prepare test data
x_test = input_envoy.Bow(test["Summary"], stop_words=None)
x_test = input_envoy.Extend_features_bow(x_train, x_test)
y_test = dataset_manager.MultiLabel2Matrix(test, genre_types)
print(y_test)
print("Test Data Ready!")
test = None
x_train = None
y_train = None


prediction = classifier.predict(x_test)
print("Test Complete!")
print(prediction)
print("hamming loss: ", metrics.hamming_loss(y_test, prediction))
print("zero_one=loss: ", metrics.zero_one_loss(y_test, prediction))
print("accuracy: ", metrics.accuracy_score(y_test, prediction))
