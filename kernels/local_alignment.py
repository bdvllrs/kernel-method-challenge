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

        self.frequencies = None
        self.couple_frequencies = None
        self.gap = lambda n: 0 if n == 0 else d + e * (n - 1)

    def score(self, x: str, y: str):
        couple_freq = 0
        if x + y in self.couple_frequencies.keys():
            couple_freq = self.couple_frequencies[x + y]
        return np.log(couple_freq / (self.frequencies[x] * self.frequencies[y]))

    def compute_frequencies(self, sequences):
        """
        Inspired from https://www.cs.cmu.edu/~02710/Lectures/ScoringMatrices2015.pdf
        and https://en.wikipedia.org/wiki/BLOSUM
        """
        self.frequencies = {"A": 0, "C": 0, "G": 0, "T": 0}
        self.couple_frequencies = {}
        total = 0
        total_couple = 0
        for sequence in sequences:
            for letter in sequence:
                self.frequencies[letter] += 1
                total += 1
            for k in range(0, len(sequence) - 1, 2):
                couple = sequence[k:k + 2]
                if couple not in self.couple_frequencies.keys():
                    self.couple_frequencies[couple] = 1
                else:
                    self.couple_frequencies[couple] += 1
                total_couple += 1
        for key in self.frequencies.keys():
            self.frequencies[key] /= total
        for key in self.couple_frequencies.keys():
            self.couple_frequencies[key] /= total_couple

    def get_M(self, i, j, x1, x2):
        if i == 0 or j == 0:
            return 0
        elif self.M[i][j] is None:
            score = np.exp(self.beta * self.score(x1[i - 1], x2[j - 1]))
            self.M[i][j] = (score * (1 + self.get_X(i - 1, j - 1, x1, x2) +
                                     self.get_Y(i - 1, j - 1, x1, x2) + self.get_M(i - 1, j - 1, x1, x2)))
        return self.M[i][j]

    def get_X(self, i, j, x1, x2):
        if i == 0 or j == 0:
            return 0
        elif self.X[i][j] is None:
            self.X[i][j] = (np.exp(self.beta * self.d) * self.get_M(i - 1, j, x1, x2)
                            + np.exp(self.beta * self.e) * self.get_X(i - 1, j, x1, x2))
        return self.X[i][j]

    def get_Y(self, i, j, x1, x2):
        if i == 0 or j == 0:
            return 0
        elif self.Y[i][j] is None:
            self.Y[i][j] = (np.exp(self.beta * self.d) * (self.get_M(i, j - 1, x1, x2) + self.get_X(i, j - 1, x1, x2))
                            + np.exp(self.beta * self.e) * self.get_Y(i, j - 1, x1, x2))
        return self.Y[i][j]

    def get_X2(self, i, j, x1, x2):
        if i == 0 or j == 0:
            return 0
        elif self.X2[i][j] is None:
            self.X2[i][j] = self.get_M(i - 1, j, x1, x2) + self.get_X2(i - 1, j, x1, x2)
        return self.X2[i][j]

    def get_Y2(self, i, j, x1, x2):
        if i == 0 or j == 0:
            return 0
        elif self.Y2[i][j] is None:
            self.Y2[i][j] = self.get_M(i, j - 1, x1, x2) + self.get_X2(i, j - 1, x1, x2) + self.get_Y2(i, j - 1, x1, x2)
        return self.Y2[i][j]

    def apply(self, embed1, embed2, _, __):
        assert self.frequencies is not None, "No sequence has been embedded before applying the kernel."
        size = len(embed1)
        self.M = [[None] * (size + 1)] * (size + 1)
        self.X = [[None] * (size + 1)] * (size + 1)
        self.Y = [[None] * (size + 1)] * (size + 1)
        self.X2 = [[None] * (size + 1)] * (size + 1)
        self.Y2 = [[None] * (size + 1)] * (size + 1)

        val = (1 + self.get_X2(size, size, embed1, embed2)
               + self.get_Y2(size, size, embed1, embed2) + self.get_M(size, size, embed1, embed2))
        val = 1 / self.beta * np.log(val)  # resole diagonal dominance issue
        print(val)
        return val

    def embed(self, sequences):
        if self.frequencies is None:
            self.compute_frequencies(sequences)
        return sequences

    def embed_one(self, sequence):
        pass
