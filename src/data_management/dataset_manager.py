import platform as pt
import pandas as pd
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


if pt.system() == "Linux":
    DATASET = "../dataset/booksummaries.txt"
elif pt.system() == "Windows":
    DATASET = "..\\dataset\\booksummaries.txt"

if pt.system() == "Linux":
    CLEANDATASET = "../dataset/booksummaries_clean.csv"
elif pt.system() == "Windows":
    CLEANDATASET = "..\\dataset\\booksummaries_clean.csv"


class DatasetManager:

    def __init__(self):
        self.data = None
        self.most_frequent_genres = {" Fiction": 0, " Novel": 0, " Fantasy": 0, " Children's literature": 0, " Mystery": 0,
                                     " Young adult literature": 0, " Suspense": 0, " Historical novel": 0, " Thriller": 0,
                                     " Horror": 0, " Romance novel": 0, " Adventure novel": 0, " Non-fiction": 0,
                                     " Alternate history": 0, " Comedy": 0}

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

    def CleanData(self):

        temp_data = self.data
        # Lower case all summaries
        temp_data["Summary"] = temp_data["Summary"].map(lambda x: x.lower())
        # Remove numbers
        temp_data["Summary"] = temp_data["Summary"].str.replace('\d+', '')
        # Remove punctuations
        temp_data["Summary"] = temp_data["Summary"].str.replace('[^\w\s]', '')
        # Filter genres
        temp_data["Genre"] = temp_data["Genre"].map(lambda x: self.SelectGenre(x))
        # Filter nan genres
        temp_data = temp_data.dropna(subset=["Genre"])
        # reset indices of rows
        temp_data = temp_data.reset_index(drop=True)
        # Save the cleaned data
        temp_data.to_csv(DATASET[0:-17] + "booksummaries_clean.csv", sep='\t', index_label=False)

        return temp_data

    def ReadCleanedData(self):

        data = pd.read_csv(CLEANDATASET, sep="\t")
        self.data = data

        return data

    def ReadCleanedDataSimplified(self):

        data = pd.read_csv(CLEANDATASET, sep="\t")
        self.data = data

        data_summaries = list(data["Summary"])
        data_genres = list(data["Genre"])

        return data_summaries, data_genres

    def SelectGenre(self, genre_list):
        if genre_list is float and math.isnan(genre_list):
            return
        # remove parenthesis
        genre_list = genre_list.strip("{}")
        # split according to comma
        genre_list = genre_list.split(",")
        true_genre = ""
        first_one_flag = 0
        for genre in genre_list:
            # remove genre code
            genre = genre.split(":")[1].replace("\"", "")
            # Change all types of fictions to " Fiction" genre
            temp_genre = genre.split(" ")
            if len(temp_genre) > 2:
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

    def SplitData(self, data=None, val_per=20, test_per=20):
        if data is None:
            data = self.data
        # Create empty dataframes
        train_data = pd.DataFrame()
        validation_data = pd.DataFrame()
        test_data = pd.DataFrame()

        data_by_genres = []
        for genre, df_genre in data.groupby('Genre'):
            # Split data by genre
            data_by_genres.append(df_genre.reset_index(drop=True))

        for df in data_by_genres:
            # Row number to be split
            val_num = int((val_per / 100) * len(df))
            test_num = int((test_per / 100) * len(df))
            # Append split data to empty dataframes
            validation_data = validation_data.append(df[0:val_num], ignore_index=True)
            test_data = test_data.append(df[val_num:val_num + test_num], ignore_index=True)
            train_data = train_data.append(df[val_num + test_num:], ignore_index=True)

        # print(len(validation_data), len(test_data), len(train_data))
        # Shuffle all data and return
        return train_data.sample(frac=1).reset_index(drop=True), \
               validation_data.sample(frac=1).reset_index(drop=True),\
               test_data.sample(frac=1).reset_index(drop=True)
