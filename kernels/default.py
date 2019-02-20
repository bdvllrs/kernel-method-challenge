import numpy as np
import hashlib

__all__ = ['Kernel', 'OneHotKernel']


class Kernel:
    def __init__(self):
        self.MEMOIZER = {}
        self.type = "linear"
        self.gamma = "auto"
        self.d = 2
        self.r = 0

    def set_args(self, kernel_type="linear", gamma="auto", degree=2, r=0):
        """
        Args:
            kernel_type:  in (linear, polynomial, gaussian, sigmoid)
            gamma: if None, auto defined
            degree: if polynomial kernel, degree of the polynomial. Default 2.
            r: added constant for polynomial and sigmoid kernel. Default 0.
        """
        self.type = kernel_type
        self.gamma = gamma
        self.d = degree
        self.r = r

    def __call__(self, data_1, data_2=None):
        """
        Returs the Gram Matrix using the embed values of the self.embed function
        Args:
            data_1: array of embeddings
            data_2: (optional) array of embeddings 2. Default: data_2 = data_1.
        """
        if data_2 is None:
            data_2 = data_1

        # Get hashes for memoization
        hash_1 = hashlib.sha256(data_1.tostring())
        hash_2 = hashlib.sha256(data_2.tostring())
        if (hash_1, hash_2) in self.MEMOIZER.keys():
            print('Using memoized data.')
            return self.MEMOIZER[(hash_1, hash_2)]
        if (hash_2, hash_1) in self.MEMOIZER.keys():
            print('Using memoized data.')
            return self.MEMOIZER[(hash_2, hash_1)].transpose()
        gram = np.array([[self.apply(x, y) for x in data_1] for y in data_2])
        self.MEMOIZER[(hash_1, hash_2)] = gram
        return gram

    def apply(self, embed1, embed2):
        """
        Returns the value of K(embed1, embed2)
        """
        if self.gamma == "auto":
            self.gamma = 1 / embed1.shape[0]
        if self.type == "polynomial":
            return self._polynomial_kernel(embed1, embed2)
        elif self.type == "gaussian":
            return self._gaussian_kernel(embed1, embed2)
        elif self.type == "sigmoid":
            return self._sigmoid_kernel(embed1, embed2)
        return self._linear_kernel(embed1, embed2)

    def _linear_kernel(self, embed1, embed2):
        return np.inner(embed1, embed2)

    def _polynomial_kernel(self, embed1, embed2):
        return (self.gamma * self._linear_kernel(embed1, embed2) + self.r) ** self.d

    def _gaussian_kernel(self, embed1, embed2):
        return np.exp(-self.gamma * np.linalg.norm(embed1 - embed2) ** 2)

    def _sigmoid_kernel(self, embed1, embed2):
        return np.tanh(self.gamma * self._linear_kernel(embed1, embed2) + self.r)

    def embed(self, sequences):
        raise NotImplemented


class OneHotKernel(Kernel):
    """
    Returns a onehot vector.
    As an example... bad in practise.
    """

    @staticmethod
    def to_onehot(sequence):
        letters = {"T": 0, "G": 1, "A": 2, "C": 3}
        onehot = np.zeros((len(sequence), 4))
        sequence = np.array([letters[letter] for letter in sequence])
        onehot[np.arange(sequence.shape[0]), sequence] = 1
        return np.ravel(onehot)

    def embed(self, sequences):
        return np.array([OneHotKernel.to_onehot(sequences[k]) for k in range(sequences.shape[0])])
