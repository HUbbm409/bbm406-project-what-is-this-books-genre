import platform as pt
import pandas as pd

if pt.system() == "Linux":
    DATASET = "../dataset/booksummaries.txt"
elif pt.system() == "Windows":
    DATASET = "..\dataset\booksummaries.txt"


class DatasetManager:

    def __init__(self):
        self.data = None

    def read_data(self):

        data = pd.read_csv(DATASET, sep="\t", header=None)
        data.columns = ["ID", "Code", "Title", "Author", "Release Date", "Genre", "Summary"]
        # Line number will be used as ID
        data = data.filter(items=["Title", "Author", "Release Date", "Genre", "Summary"])
        # Filter nan genres
        data = data.dropna(subset=["Genre"])
        # reset indices of rows
        data = data.reset_index(drop=True)

        self.data = data
        return data
