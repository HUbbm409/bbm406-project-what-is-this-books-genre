from data_management.dataset_manager import DatasetManager
import matplotlib


manager = DatasetManager()
manager.ReadData()
manager.CleanData()
manager.ReadCleanedData()
manager.SplitData(val_per=20, test_per=10)
