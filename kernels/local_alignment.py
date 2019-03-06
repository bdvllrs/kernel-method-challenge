__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
from kernels.default import Kernel

__all__ = ['LocalAlignmentKernel']


class LocalAlignmentKernel(Kernel):
    """
    Local Alignment Kernel.
    Inspired from http://members.cbio.mines-paristech.fr/~jvert/publi/04kmcbbook/saigo.pdf
    Vert et al. 2004
    """

    def __init__(self, memoize_conf, beta, d, e):
        """
        Args:
            memoize_conf:
            beta:
            d, e: constant for the gap
        """
        super(LocalAlignmentKernel, self).__init__(memoize_conf)

        self.d = d
        self.e = e
        self.beta = beta

        self.M = None
        self.X = None
        self.Y = None
        self.X2 = None
        self.Y2 = None

        self.S = None
        self.gap = lambda n: 0 if n == 0 else d + e * (n - 1)

    def score(self, x: str, y: str):
        if not x + y in self.S.keys():
            return 0
        return self.S[x + y]

    def compute_frequencies(self, sequences):
        """
        Inspired from https://www.cs.cmu.edu/~02710/Lectures/ScoringMatrices2015.pdf
        and https://en.wikipedia.org/wiki/BLOSUM
        """
        frequencies = {"A": 0, "C": 0, "G": 0, "T": 0}
        couple_frequencies = {}
        self.S = {}
        total = 0
        total_couple = 0
        for sequence in sequences:
            for letter in sequence:
                frequencies[letter] += 1
                total += 1
            for k in range(0, len(sequence) - 1, 2):
                couple = sequence[k:k + 2]
                if couple not in couple_frequencies.keys():
                    couple_frequencies[couple] = 1
                else:
                    couple_frequencies[couple] += 1
                total_couple += 1
        for key in frequencies.keys():
            frequencies[key] /= total
        for key in couple_frequencies.keys():
            couple_frequencies[key] /= total_couple
            self.S[key] = np.log(couple_frequencies[key] / (frequencies[key[0]] * frequencies[key[1]]))

    def apply(self, x1, x2, _, __):
        size = len(x1) + 1
        M = np.zeros((size, size))
        X = np.zeros((size, size))
        Y = np.zeros((size, size))
        X2 = np.zeros((size, size))
        Y2 = np.zeros((size, size))

        exp_beta_d = np.exp(self.beta * self.d)
        exp_beta_e = np.exp(self.beta * self.e)

        for i in range(1, size):
            for j in range(1, size):
                score = np.exp(self.beta * self.score(x1[i - 1], x2[j - 1]))
                M[i, j] = score * (1 + X[i - 1, j - 1] + Y[i - 1, j - 1] + M[i - 1, j - 1])
                X[i, j] = exp_beta_d * M[i - 1, j] + exp_beta_e * X[i - 1, j]
                Y[i, j] = exp_beta_d * (M[i, j - 1] + X[i, j - 1]) + exp_beta_e * Y[i, j - 1]
                X2[i, j] = M[i - 1, j] + X2[i - 1, j]
                Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]

        val = 1 + X2[size - 1, size - 1] + Y2[size - 1, size - 1] + M[size - 1, size - 1]
        return 1 / self.beta * np.log(val)  # resole diagonal dominance issue

    def embed(self, sequences):
        if self.S is None:
            self.compute_frequencies(sequences)
        return sequences

    def embed_one(self, sequence):
        pass
