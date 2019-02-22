import numpy as np
from kernels.default import Kernel
from itertools import product


__all__ = ['SpectrumKernel']


class Mismatch(Kernel):


    def __init__(self, length = 3, n_sample= 7):
        super(Mismatch, self).__init__()
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
            X_train_histo_0[seq_idx, :] = self.embed_one(seq_val, self.n)

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

    def gram_matrix(self, X1, X2):
        len_X2 = len(X2)
        len_X1 = len(X1)
        sim_docs_kernel_value = {}
        if len_X2 == 0:
            gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)
            for i in range(len_X1):
                sim_docs_kernel_value[i] = compute_diag_copy(X1)(i)

            for i in range(len_X1):
                for j in range(i, len_X1):
                    gram_matrix[i, j] = compute_element_i(X1, sim_docs_kernel_value, i)(j)
                    gram_matrix[j, i] = gram_matrix[i, j]
            # calculate Gram matrix
            return gram_matrix
        else:
            gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
            for i in range(len_X1):
                sim_docs_kernel_value[1][i] = compute_diag_copy(X1)(i)
            for j in range(len_X2):
                sim_docs_kernel_value[2][j] = compute_diag_copy(X2)(j)

            for i in range(len_X1):
                for j in range(len_X2):
                    gram_matrix[i, j] = compute_element_i_general(X1, X2, sim_docs_kernel_value, i)(j)
            return gram_matrix


compute_diag = lambda X,i: np.vdot(X[i], X[i])
compute_element_kernel_square = lambda X1,sim_docs_kernel_value,i,j: np.vdot(X1[i], X1[j])/(sim_docs_kernel_value[i] *sim_docs_kernel_value[j])**0.5
compute_element_kernel = lambda X1,X2,sim_docs_kernel_value,i,j: np.vdot(X1[i], X2[j])/(sim_docs_kernel_value[1][i] *sim_docs_kernel_value[2][j])**0.5


class compute_diag_copy(object):
    def __init__(self, X):
        self.X = X
    def __call__(self, i):
        return compute_diag(self.X,i)

class compute_element_i(object):
    def __init__(self, X,sim_docs_kernel_value,i):
        self.X = X
        self.sim_docs_kernel_value = sim_docs_kernel_value
        self.i = i
    def __call__(self, j):
        return compute_element_kernel_square(self.X,self.sim_docs_kernel_value,self.i,j)

class compute_element_i_general(object):
    def __init__(self, X,X_p,sim_docs_kernel_value,i):
        self.X = X
        self.X_p = X_p
        self.sim_docs_kernel_value = sim_docs_kernel_value
        self.i = i
    def __call__(self, j):
        return compute_element_kernel(self.X,self.X_p,self.sim_docs_kernel_value,self.i,j)