__author__ = "Benjamin Devillers (bdvllrs)"
__version__ = "1.0.0"
__all__ = ["Memoizer"]

import os
import pickle


class Memoizer:
    def __init__(self, conf, name):
        self._MEMOIZER = None
        self.path = os.path.abspath(os.path.join(os.curdir, conf.path))
        self.file_path = os.path.join(self.path, name + "Data.pkl")

    def load(self):
        # print("Loading memoized data...")
        # self._MEMOIZER = {}
        # if self.file_path in os.listdir(self.path):
        #     with open(self.file_path, 'rb') as f:
        #         self._MEMOIZER = pickle.load(f)
        # print("Loaded.")
        pass

    def save(self):
        # print("Saving memoized data...")
        # with open(self.file_path, 'wb') as f:
        #     pickle.dump(self._MEMOIZER, f)
        # print("Saved.")
        pass

    def add(self, path, value):
        """
        Add data to the memoizer
        Args:
            path: path of the data, can be separated by points. e.g. name = "idx1.idx2" will get data in store["idx1"]["idx2"].
            value:  value to store
        """
        # if self._MEMOIZER is None:
        #     self.load()
        # self._MEMOIZER[path] = value
        pass

    def __setitem__(self, key, value):
        self.add(key, value)

    def get(self, path, default=None):
        """
        Retrieve data from the memoizer
        Args:
            path: path of the data, can be separated by points. e.g. name = "idx1.idx2" will get data in store["idx1"]["idx2"].
            default: Default value if not in the memoizer
        """
        return None
        # if self._MEMOIZER is None:
        #     self.load()
        # if path in self._MEMOIZER.keys():
        #     return self._MEMOIZER[path]
        # return default

    def __getitem__(self, key):
        return self.get(key)

    def is_in(self, path):
        """
        Check if an index is in the memoizer
        """
        # if self._MEMOIZER is None:
        #     self.load()
        # if path in self._MEMOIZER.keys():
        #     return True
        return False

    def __contains__(self, item):
        return self.is_in(item)
