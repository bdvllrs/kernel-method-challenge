__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
from kernels.default import Kernel

__all__ = ['LocalAlignmentKernel']


class LocalAlignmentKernel(Kernel):

    def __init__(self, memoize_conf, d, e):
        """
        Args:
            memoize_conf:
            d, e: constant for the gap
        """
        super(LocalAlignmentKernel, self).__init__(memoize_conf)

        self.d = d
        self.e = e
        if "la.M" in self.memoizer:
            self.M = self.memoizer["la.M"]
            self.X = self.memoizer["la.X"]
            self.Y = self.memoizer["la.Y"]
            self.X2 = self.memoizer["la.X2"]
            self.X2 = self.memoizer["la.X2"]
        self.gap = lambda n: 0 if n == 0 else d + e * (n - 1)

    def apply(self, embed1, embed2):
        pass

    def embed(self, sequences):
        return sequences

    def embed_one(self, sequence):
        pass
