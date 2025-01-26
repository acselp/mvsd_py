import os
import pandas as pd
from config import DATA_DIR


class FileProvider:
    def __init__(self):
        self._data_dir = DATA_DIR
        self._test_data_name = 'test.csv'
        self._train_data_name = 'train.csv'

    def _get_data_file(self, filename) -> str | bytes:
        return os.path.join(self._data_dir, filename)

    def get_train_data(self):
        return pd.read_csv(self._get_data_file(self._train_data_name))

    def get_test_data(self):
        return pd.read_csv(self._get_data_file(self._test_data_name))