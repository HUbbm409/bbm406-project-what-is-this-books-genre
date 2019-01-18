from data_management.dataset_manager import DatasetManager
import matplotlib.pyplot as plt
import ast
import numpy as np

genre_types = {' Fiction': 0, ' Speculative fiction': 0, ' Science Fiction': 0, ' Novel': 0, ' Fantasy': 0,
                 " Children's literature": 0, ' Mystery': 0, ' Young adult literature': 0, ' Suspense': 0,
                 ' Crime Fiction': 0, ' Historical novel': 0, ' Thriller': 0, ' Horror': 0, ' Romance novel': 0,
                 ' Historical fiction': 0, ' Detective fiction': 0, ' Adventure novel': 0, ' Non-fiction': 0,
                 ' Alternate history': 0, ' Spy fiction': 0, ' Comedy': 0, ' Dystopia': 0, ' Autobiography': 0,
                 ' Satire': 0, ' Gothic fiction': 0, ' Comic novel': 0, ' Biography': 0}


manager = DatasetManager()
# manager.ReadData()
# manager.CleanData(multi_genre=True)
manager.ReadCleanedData()

data = manager.data["GenreList"]

label_numbers = {}

for genres in data:
    genres = ast.literal_eval(genres)

    if len(genres) not in label_numbers:
        label_numbers[len(genres)] = 0

    label_numbers[len(genres)] += 1

    for genre in genres:
        genre_types[genre] += 1

print(genre_types)

label_numbers = sorted(label_numbers.items(), key=lambda kv: kv[0])
print(label_numbers)

plt.bar(genre_types.keys(), genre_types.values())
plt.xticks(np.arange(len(genre_types.keys())), genre_types.keys(), rotation='vertical')
plt.title("Label Distribution")
plt.ylabel("# Number of Occurrences")
plt.xlabel("Label Name")
plt.show()

x = [a[0] for a in label_numbers]
y = [a[1] for a in label_numbers]

plt.bar(x, y)
plt.title("Number of labels for each book")
plt.ylabel("# Number of Occurrences")
plt.xlabel("# Number of Labels")
plt.show()