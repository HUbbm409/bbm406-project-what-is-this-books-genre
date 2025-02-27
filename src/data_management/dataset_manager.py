import platform as pt
import pandas as pd
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import ast
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse

if pt.system() == "Linux":
    DATASET = "../../dataset/booksummaries.txt"
elif pt.system() == "Windows":
    DATASET = "..\\..\\dataset\\booksummaries.txt"

if pt.system() == "Linux":
    CLEANDATASET = "../../dataset/booksummaries_clean.csv"
elif pt.system() == "Windows":
    CLEANDATASET = "..\\..\\dataset\\booksummaries_clean.csv"


class DatasetManager:

    def __init__(self):
        self.data = None
        self.most_frequent_genres = {' Fiction': 0, ' Speculative fiction': 0, ' Science Fiction': 0, ' Novel': 0, ' Fantasy': 0,
                                     " Children's literature": 0, ' Mystery': 0, ' Young adult literature': 0, ' Suspense': 0,
                                     ' Crime Fiction': 0, ' Historical novel': 0, ' Thriller': 0, ' Horror': 0, ' Romance novel': 0,
                                     ' Historical fiction': 0, ' Detective fiction': 0, ' Adventure novel': 0, ' Non-fiction': 0,
                                     ' Alternate history': 0, ' Spy fiction': 0, ' Comedy': 0, ' Dystopia': 0, ' Autobiography': 0,
                                     ' Satire': 0, ' Gothic fiction': 0, ' Comic novel': 0, ' Biography': 0}

    def ReadData(self):

        data = pd.read_csv(DATASET, sep="\t", header=None)
        data.columns = ["ID", "Code", "Title", "Author", "Release_Date", "Genre", "Summary"]
        # Line number will be used as ID
        data = data.filter(items=["Title", "Author", "Release_Date", "Genre", "Summary"])
        # Filter nan genres
        data = data.dropna(subset=["Genre"])
        # reset indices of rows
        data = data.reset_index(drop=True)

        self.data = data
        return data

    def CleanData(self, multi_genre=False):

        temp_data = self.data
        # Lower case all summaries
        temp_data["Summary"] = temp_data["Summary"].map(lambda x: x.lower())
        # Remove numbers
        temp_data["Summary"] = temp_data["Summary"].str.replace('\d+', '')
        # Remove punctuations
        temp_data["Summary"] = temp_data["Summary"].str.replace('[^\w\s]', '')
        # Filter genres
        if not multi_genre:
            temp_data["Genre"] = temp_data["Genre"].map(lambda x: self.SelectGenre(x))
        else:
            temp_data["GenreList"] = temp_data["Genre"].map(lambda x: self.SelectGenreMultipleGenre(x))
            temp_data["Genre"] = temp_data["Genre"].map(lambda x: self.SelectGenre(x, multi_genre=True))
        # Filter nan genres
        temp_data = temp_data.dropna(subset=["Genre"])
        # reset indices of rows
        temp_data = temp_data.reset_index(drop=True)
        # Save the cleaned data
        temp_data.to_csv(DATASET[0:-17] + "booksummaries_clean.csv", sep='\t', index_label=False)

        return temp_data

    def ReadCleanedData(self):
        # Read whole dataset and return it
        data = pd.read_csv(CLEANDATASET, sep="\t")
        self.data = data

        return data

    def ReadCleanedDataSimplified(self):
        # Read whole dataset, but return only summary and genre columns
        data = pd.read_csv(CLEANDATASET, sep="\t")
        self.data = data

        data_summaries = list(data["Summary"])
        data_genres = list(data["Genre"])

        return data_summaries, data_genres

    def SelectGenre(self, genre_list, multi_genre=False):
        if genre_list is float and math.isnan(genre_list):
            return
        # Remove parenthesis
        genre_list = genre_list.strip("{}")
        # Split according to comma
        genre_list = genre_list.split(",")
        true_genre = ""
        first_one_flag = 0
        for genre in genre_list:
            # Remove genre code
            genre = genre.split(":")[1].replace("\"", "")
            # Change all types of fictions to " Fiction" genre
            temp_genre = genre.split(" ")
            if len(temp_genre) > 2 and not multi_genre:
                if temp_genre[2] == "fiction" or temp_genre[2] == "Fiction":
                    genre = " Fiction"

            mfg = self.most_frequent_genres

            if genre in mfg:
                # These are done to have a more homogeneous dataset
                # Check if the genre is the first one to meet the conditions
                if first_one_flag == 0:
                    true_genre = genre
                    first_one_flag = 1
                # If genre count is lower than true_genre count, update true_genre
                if mfg[genre] <= mfg[true_genre]:
                    true_genre = genre
        # Check if algorithm found a genre
        if true_genre != "":
            self.most_frequent_genres[true_genre] += 1
        else:
            return
        # Update book's genre in dataset
        return true_genre

    def SelectGenreMultipleGenre(self, genre_list):
        if genre_list is float and math.isnan(genre_list):
            return
        # Remove parenthesis
        genre_list = genre_list.strip("{}")
        # Split according to comma
        genre_list = genre_list.split(",")
        true_genre = []

        for genre in genre_list:
            # Remove genre code
            genre = genre.split(":")[1].replace("\"", "")
            # Append genres to list
            if genre in self.most_frequent_genres:
                true_genre.append(genre)

        if true_genre is []:
            return
        else:
            return true_genre


    def Lemmatize(self, data=None):
        if data is None:
            data = self.data["Summary"]

        # Initialize lemmatizer
        nltk.download("wordnet")
        lemmer = WordNetLemmatizer()
        # Lemmatize summaries
        data = [' '.join([lemmer.lemmatize(word) for word in summary.split(' ')]) for summary in data]

        return data

    def Stemmize(self, data=None):
        if data is None:
            data = self.data["Summary"]
        # Initialize stemmer
        stemmer = SnowballStemmer('english')
        # Stem the data
        data = [' '.join([stemmer.stem(word) for word in summary.split(' ')]) for summary in data]

        return data

    def SplitData(self, data=None, test_per=20):
        if data is None:
            data = self.data
        # Create empty dataframes
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        data_by_genres = []
        for genre, df_genre in data.groupby('Genre'):
            # Split data by genre
            data_by_genres.append(df_genre.reset_index(drop=True))

        for df in data_by_genres:
            # Row number to be split
            test_num = int((test_per / 100) * len(df))
            # Append split data to empty dataframes
            test_data = test_data.append([df[0:test_num]], ignore_index=True)
            train_data = train_data.append([df[test_num:]], ignore_index=True)

        # print(len(validation_data), len(test_data), len(train_data))
        # Shuffle all data and return
        return train_data.sample(frac=1).reset_index(drop=True), \
               test_data.sample(frac=1).reset_index(drop=True)

    def SplitDataMultipleGenre(self, data=None, test_per=20):
        # Check if data is given
        if data is None:
            data = self.data
        # Test data percentage
        test_num = test_per/100
        # Split dataset
        train_data, test_data = train_test_split(data, test_size=test_num, shuffle=True)
        # Create GenreList for each book
        genre_set = set()

        for genres in data["GenreList"]:
            genres = ast.literal_eval(genres)
            for genre in genres:
                genre_set.add(genre)

        return train_data, test_data, genre_set

    def MultiLabel2Matrix(self, data, genre_types):
        # Check if whole dataset or not
        if isinstance(data, pd.DataFrame):
            data = data["GenreList"]
        y = []
        for genres in data:
            genres = ast.literal_eval(genres)
            # Create a vector with length of genre type count
            temp = [0] * len(genre_types)
            for genre in genres:
                # Fill the vector
                temp[genre_types[genre]] = 1
            # Add to y
            y.append(temp)
        # Change type to csr_matrix
        outputs = np.array(y)
        y = sparse.lil_matrix(outputs)

        return y
