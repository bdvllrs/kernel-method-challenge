__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
from kernels.default import Kernel

__all__ = ['SpectrumKernel']


class SpectrumKernel(Kernel):
    """
    Inspired from slide 55 of http://members.cbio.mines-paristech.fr/~jvert/talks/060727mlss/mlss.pdf
    """

    def __init__(self, memoize_conf, length=3):
        super(SpectrumKernel, self).__init__(memoize_conf)
        self.length = length
        self.possible_nuples = {}

    def config_digest(self):
        return f"length-{self.length}"

    def add_nuples(self, sequence):
        for k in range(0, len(sequence) - (self.length - 1)):
            nuple = sequence[k:k + self.length]
            if nuple in self.possible_nuples.keys():
                self.possible_nuples[nuple] += 1
            else:
                self.possible_nuples[nuple] = 1
        return self.possible_nuples

    def embed_one(self, sequence):
        nuples = {nuple: k for k, nuple in enumerate(self.possible_nuples.keys())}
        vec = np.zeros(len(nuples.keys()))
        for k in range(0, len(sequence) - (self.length - 1)):
            nuple = sequence[k:k + self.length]
            if nuple in nuples.keys():
                vec[nuples[nuple]] += 1
        return vec

    def embed(self, sequences):
        if "nuples" in self.memoizer:
            self.possible_nuples = self.memoizer["nuples"]
        elif self.possible_nuples == {}:
            for sequence in sequences:
                self.add_nuples(sequence)
            self.memoizer["nuples"] = self.possible_nuples
        return super(SpectrumKernel, self).embed(sequences)

    # def apply(self, embed1, embed2, _, __):
    #     total = 0
    #     for embed in embed1.keys():
    #         if embed in embed2.keys():
    #             total += embed1[embed] * embed2[embed]
    #     return total
