import numpy as np
from kernels.default import Kernel

__all__ = ['SpectrumKernel']


class SpectrumKernel(Kernel):
    """
    Inspired from slide 55 of http://members.cbio.mines-paristech.fr/~jvert/talks/060727mlss/mlss.pdf
    """

    def __init__(self, length=3):
        super(SpectrumKernel, self).__init__()
        self.length = length

    def embed_one(self, sequence):
        all_nuples = {}
        for k in range(len(sequence) - (self.length - 1)):
            nuple = sequence[k:k + self.length]
            if nuple in all_nuples.keys():
                all_nuples[nuple] += 1
            else:
                all_nuples[nuple] = 1
        return all_nuples

    def embed(self, sequences):
        return np.array([self.embed_one(sequences[k]) for k in range(sequences.shape[0])])

    def apply(self, embed1, embed2):
        total = 0
        for embed in embed1.keys():
            if embed in embed2.keys():
                total += embed1[embed] * embed2[embed]
        return total
