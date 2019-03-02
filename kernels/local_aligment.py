__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
from kernels.default import Kernel

__all__ = ['LocalAlignmentKernel']


class LocalAlignmentKernel(Kernel):

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

        self.gap = lambda n: 0 if n == 0 else d + e * (n - 1)
        self.S = lambda x, y: 1 if x == y else 0

    def get_M(self, i, j, x1, x2):
        if i == 0 or j == 0:
            return 0
        elif self.M[i][j] is None:
            score = np.exp(self.beta * self.S(x1[i - 1], x2[j - 1]))
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
        size = len(embed1)
        self.M = [[None] * (size + 1)] * (size + 1)
        self.X = [[None] * (size + 1)] * (size + 1)
        self.Y = [[None] * (size + 1)] * (size + 1)
        self.X2 = [[None] * (size + 1)] * (size + 1)
        self.Y2 = [[None] * (size + 1)] * (size + 1)

        val = (1 + self.get_X2(size, size, embed1, embed2)
               + self.get_Y2(size, size, embed1, embed2) + self.get_M(size, size, embed1, embed2))
        print(val)
        return val

    def embed(self, sequences):
        return sequences

    def embed_one(self, sequence):
        pass
