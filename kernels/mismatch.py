__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
from itertools import product
from kernels.default import Kernel

__all__ = ['MismatchKernel']


def hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


class MismatchKernel(Kernel):

    def __init__(self, memoize_conf, k=3, m=2):
        super(MismatchKernel, self).__init__(memoize_conf)
        self.k = k
        self.m = m
        self.possible_mismatches = {}

    def config_digest(self):
        return f"k-{self.k}-m-{self.m}"

    def add_mer(self, sequence):
        """
        Pre-indexation of the k-mer
        Args:
            sequence:
        """
        for k in range(0, len(sequence) - (self.k - 1)):
            mer = sequence[k:k + self.k]
            self.possible_mismatches[mer] = []

    def compute_mismatch_sequences(self):
        all_mers = list(product("ATGC", repeat=self.k))
        for mer1 in self.possible_mismatches.keys():
            for mer2 in all_mers:
                if hamming_distance(mer1, ''.join(mer2)) <= self.m:
                    self.possible_mismatches[mer1].append(''.join(mer2))

    def mer_to_idx(self, seq):
        letter_to_number = {"A": 0, "G": 1, "C": 2, "T": 3}
        idx = 0
        factor = 1
        for letter in seq:
            idx += letter_to_number[letter] * factor
            factor *= 4
        return idx

    def embed_one(self, sequence):
        """
        Embed one sequence into the embedding
        Args:
            sequence:
        """
        vec = np.zeros(4 ** self.k)  # all possible combination of AGTC
        for k in range(0, len(sequence) - (self.k - 1)):  # look for all mers in the sequence
            mer = sequence[k:k + self.k]
            for mismatch in self.possible_mismatches[mer]:  # go through all mismatches of the mer
                idx = self.mer_to_idx(mismatch)
                vec[idx] += 1
        return vec

    def embed(self, sequences):
        if self.possible_mismatches == {}:
            print("Add all mer sequences")
            for sequence in sequences:
                self.add_mer(sequence)
            print("Compute the mismatches possibilities")
            self.compute_mismatch_sequences()
            print("Saving into memoizer.")
            self.memoizer["nuples"] = self.possible_mismatches
        return super(MismatchKernel, self).embed(sequences)

    # def apply(self, embed1, embed2, _, __):
    #     total = 0
    #     for embed in embed1.keys():
    #         if embed in embed2.keys():
    #             total += embed1[embed] * embed2[embed]
    #     return total
