import numpy as np
from kernels.default import Kernel
from itertools import product



__all__ = ['MismatchKernel']


class MismatchKernel(Kernel):


    def __init__(self, memoize_conf, length = 3, n_sample= 7):
        super(MismatchKernel, self).__init__(memoize_conf)
        self.length = length
        self.n_sample = n_sample
        self.possible_numples = {}
        self.letters = ['A', 'C', 'G', 'T']
        self.ngrams = lambda a, n: list(zip(*[a[i:] for i in range(n)]))
        self.all_combin_list = self.create_all_possible_combination(n_sample)

    def create_all_possible_combination(self, n):
        return list(product(self.letters, repeat=n))

    def embed(self, sequences):
        X_train_histo_0 = np.empty([len(sequences), len(self.all_combin_list)])
        for seq_idx, seq_val in enumerate(sequences):
            X_train_histo_0[seq_idx, :] = self.embed_one(seq_val, self.n_sample)
        return(X_train_histo_0)

    def embed_one(self, sequence, n):
        decompose_seq = self.ngrams(sequence, n)
        value = np.zeros([len(self.all_combin_list), ])
        for ngram in decompose_seq:
            index_ngram = self.all_combin_list.index(ngram)
            value[index_ngram] = value[index_ngram] + 1
            copy_ngram = list(ngram)
            for ind, cur_letter in enumerate(copy_ngram):
                for letter in self.letters:
                    if letter != cur_letter:
                        new_ngram = list(copy_ngram)
                        new_ngram[ind] = letter
                        mismatch_ngram = tuple(new_ngram)
                        index_ngram = self.all_combin_list .index(mismatch_ngram)
                        value[index_ngram] = value[index_ngram] + 0.1
        return value