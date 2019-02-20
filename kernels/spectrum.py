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
        self.possible_numples = {}

    def add_nuples(self, sequence):
        for k in range(0, len(sequence) - (self.length - 1), self.length):
            nuple = sequence[k:k + self.length]
            if nuple in self.possible_numples.keys():
                self.possible_numples[nuple] += 1
            else:
                self.possible_numples[nuple] = 1
        return self.possible_numples

    def embed_one(self, sequence):
        nuples = {nuple: k for k, nuple in enumerate(self.possible_numples.keys())}
        vec = np.zeros(len(nuples.keys()))
        for k in range(0, len(sequence) - (self.length - 1), self.length):
            nuple = sequence[k:k + self.length]
            if nuple in nuples.keys():
                vec[nuples[nuple]] += 1
        return vec

    def embed(self, sequences):
        if self.possible_numples == {}:
            for sequence in sequences:
                self.add_nuples(sequence)
        return np.array([self.embed_one(sequences[k]) for k in range(sequences.shape[0])])

    # def apply(self, embed1, embed2):
    #     total = 0
    #     for embed in embed1.keys():
    #         if embed in embed2.keys():
    #             total += embed1[embed] * embed2[embed]
    #     return total
