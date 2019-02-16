import numpy as np
from pandas.core.util.hashing import hash_pandas_object

__all__ = ['Kernel', 'OneHotKernel']


class Kernel:
    def __init__(self):
        self.MEMOIZER = {}

    def __call__(self, data_1, data_2=None):
        """
        Returs the Gram Matrix using the embed values of the self.embed function
        Args:
            data_1: DataFrame containing vec
            data_2: (optional) DataFrame containing vec. Default: data_2 = data_1.
        """
        if data_2 is None:
            data_2 = data_1

        # Get hashes for memoization
        hash_1 = hash_pandas_object(data_1['seq']).sum()
        hash_2 = hash_pandas_object(data_2['seq']).sum()
        if (hash_1, hash_2) in self.MEMOIZER.keys():
            return self.MEMOIZER[(hash_1, hash_2)]
        if (hash_2, hash_1) in self.MEMOIZER.keys():
            return self.MEMOIZER[(hash_2, hash_1)].transpose()
        gram = np.array([[self.apply(x, y) for x in data_1['vec'].values] for y in data_2['vec'].values])
        self.MEMOIZER[(hash_1, hash_2)] = gram
        return gram

    def apply(self, embed1, embed2):
        """
        Returns the value of K(embed1, embed2)
        """
        return np.inner(embed1, embed2)

    def embed(self, sequence):
        raise NotImplemented


class OneHotKernel(Kernel):
    """
    Returns a onehot vector.
    As an example... bad in practise.
    """

    def embed(self, sequence):
        letters = {"T": 0, "G": 1, "A": 2, "C": 3}
        onehot = np.zeros((len(sequence), 4))
        sequence = np.array([letters[letter] for letter in sequence])
        onehot[np.arange(sequence.shape[0]), sequence] = 1
        return np.ravel(onehot)


class Combine(Kernel):
    # TODO: Combining kernel
    pass
