import platform as pt
import pandas as pd
import string

if pt.system() == "Linux":
    DATASET = "../dataset/booksummaries.txt"
elif pt.system() == "Windows":
    DATASET = "..\\dataset\\booksummaries.txt"


class DatasetManager:

    def __init__(self):
        self.data = None

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

        temp_data.to_csv(DATASET[0:-17] + "booksummaries_clean.csv", sep='\t')

        return temp_data
