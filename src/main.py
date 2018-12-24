from data_management.dataset_manager import DatasetManager
import matplotlib


manager = DatasetManager()
manager.ReadData()
manager.CleanData(multi_genre=True)
manager.ReadCleanedData()
