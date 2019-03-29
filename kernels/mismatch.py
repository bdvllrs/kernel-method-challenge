__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
from itertools import product
from kernels.default import Kernel
from utils.config import Config

__all__ = ['MismatchKernel']


def hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


class MismatchKernel(Kernel):
    """
    Mismatch kernel from Leslie et al. (https://cseweb.ucsd.edu/~eeskin/papers/mismatch-nips02.pdf)

    Implementation with pre-indexing.
    This can cause a memory issue if k is too big.
    We only experiment with k <= 6, and pre-indexing is possible.
    """

    def __init__(self, memoize_conf, k=3, m=2):
        """
        Args:
            memoize_conf:
            k: size of the mers
            m: number of mismatches allowed
        """
        super(MismatchKernel, self).__init__(memoize_conf)
        self.k = k
        self.m = m
        self.possible_mismatches = {}
        self.all_mers = None

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
        # product returns a tuple of letter ('A', ..., 'C')
        # we apply a map to get the string out of the tuple
        self.all_mers = list(map(lambda x: ''.join(x), product("ATGC", repeat=self.k)))
        for mer1 in self.possible_mismatches.keys():
            for mer2 in self.all_mers:
                if hamming_distance(mer1, mer2) <= self.m:
                    self.possible_mismatches[mer1].append(mer2)
        self.all_mers = {mer: k for k, mer in enumerate(self.all_mers)}

    def embed_one(self, sequence):
        """
        Embed one sequence into the embedding
        Args:
            sequence:
        """
        vec = np.zeros(4 ** self.k)  # all possible combination of AGTC
        for k in range(0, len(sequence) - (self.k - 1)):  # look for all mers in the sequence
            mer = sequence[k:k + self.k]
            if mer in self.possible_mismatches.keys():
                for mismatch in self.possible_mismatches[mer]:  # go through all mismatches of the mer
                    vec[self.all_mers[mismatch]] += 1
        return vec

    def embed(self, sequences):
        """
        Pre-index possible k-mers then compute embedding of all sequences
        """
        # pre-index possible k-mers for the training set, not for testing
        if self.possible_mismatches == {}:
            print("Add all mer sequences")
            for sequence in sequences:
                self.add_mer(sequence)
            print("Compute the mismatches possibilities")
            self.compute_mismatch_sequences()
            print("Embed sequences")
        # Compute embeddings
        return super(MismatchKernel, self).embed(sequences)


if __name__ == "__main__":
    config = Config("../config/")
    kernel = MismatchKernel(config["global"].kernels.memoize, 2, 1)
    print(kernel.embed(np.array(["ACCG"])))
    print(kernel.possible_mismatches)
    print(kernel.all_mers)

