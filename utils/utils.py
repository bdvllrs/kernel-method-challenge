import pandas as pd
import numpy as np
import random
import os
import kernels
import classifiers

__all__ = ['get_classifier', 'get_kernel', 'get_sets', 'split_train_val']


def get_sets(path, slug="tr"):
    path = os.path.abspath(os.path.join(os.curdir, path))
    data = []
    for k in range(3):
        data.append(pd.read_csv(os.path.join(path, "X{}{}.csv".format(slug, k))))
        if slug == "tr":
            labels = pd.read_csv(os.path.join(path, "Y{}{}.csv".format(slug, k)))
            data[-1] = data[-1].merge(labels, on="Id")
    # data = pd.concat(data)
    # data["vec"] = data.apply(lambda x: kernel.embed(x['seq']), axis=1)
    return data


def split_train_val(data, ratio):
    shuffled_data = data.sample(frac=1)
    last_key = int(ratio * data.shape[0])
    return shuffled_data[:last_key], shuffled_data[last_key:]


def get_kernel(kernel: str, args) -> kernels.Kernel:
    kernel = kernel.lower()
    assert kernel in ['onehot', 'spectrum'], "Unknown requested kernel."

    if kernel == "spectrum":
        default_args = {"length": 3}
        default_args.update(args)
        return kernels.SpectrumKernel(default_args['length'])
    return kernels.OneHotKernel()


def get_classifier(classifier: str, training_data, kernel, args) -> classifiers.Classifier:
    classifier = classifier.lower()
    assert classifier in ['svm'], "Unknown requested classifier."

    if classifier == "svm":
        assert "lbd" in args.keys(), "ldb must be in config.classifiers.args for svm."
        return classifiers.SVMClassifier(training_data, kernel, args['lbd'])


# A=(1, 0, 0, 0), C=(0, 1, 0, 0), G=(0, 0, 1, 0), T=(0, 0, 0, 1)

def transform_letter_in_one_hot_vector(letter):
    '''
    Compute the function that transform a letter into its one hot vector equivalent vector
    Param: @letter : (str) a letter within A,C,G,T
    Return: (list) with the one hot embedding representation of the letter given as input
    '''
    if letter == 'A':
        return [1, 0, 0, 0]
    elif letter == 'C':
        return [0, 1, 0, 0]
    elif letter == 'G':
        return [0, 0, 1, 0]
    elif letter == 'T':
        return [0, 0, 0, 1]


def transform_seq_into_spare_hot_vector(sequence):
    '''
    Transform a all sequence into its one hot vector equivalent vector (concatenation of the one hot representation of each letter in the sequence)
    Param: @sequence : (str) a sequence of strings within A,C,G,T
    Return: (list) representation of the sequence
    '''
    vector = [transform_letter_in_one_hot_vector(letter) for letter in sequence]
    vector = np.array(vector).reshape(-1)
    return vector.tolist()


def transform_seq_into_spare_hot_vector_hmm(sequence):  # almost the same as the previously described function
    '''
    Transform a all sequence into its one hot vector equivalent vector (concatenation of the one hot representation of each letter in the sequence)
    Param: @sequence : (str) a sequence of strings within A,C,G,T
    Return: (np.array) representation of the sequence
    '''
    vector = [transform_letter_in_one_hot_vector(letter) for letter in sequence]
    return np.array(vector)


def transform_data_into_sparse_hot_vector(data_seq_matrix):
    '''
    Transform a full list of sequences into their one hot vector equivalent vector (concatenation of the one hot representation of each letter in the sequence)
    Param: @data_seq_matrix : (list) list of sequences
    Return: (np.array)representation of each sequences
    '''
    matrix = [transform_seq_into_spare_hot_vector(seq[0]) for seq in data_seq_matrix]
    matrix = np.array(matrix)
    return matrix


def transform_seq_into_label_encode(sequence):
    '''
    Label encoder for our DNA sequences
    '''
    transform = lambda letter: 0 if letter == 'A' else 1 if letter == 'C' else 2 if letter == 'G' else 3
    vector = [transform(letter) for letter in sequence]
    return np.array(vector)


def compute_patch(sequence, nb_cut):
    '''
    Function that computes multiple patches representing the sequence
    Param: @sequence: (str) a DNA sequence of strings within A,C,G,T
    @nb_cut: (int) number of patches
    '''
    one_hot_vect = transform_seq_into_spare_hot_vector_hmm(sequence).reshape(-1)
    len_one_patch = int(len(one_hot_vect) / nb_cut)
    if len_one_patch * nb_cut != len(one_hot_vect):
        nb_rest = len(one_hot_vect) % (len_one_patch * nb_cut)
        one_hot_vect = np.concatenate((one_hot_vect, np.zeros(nb_rest)))
        len_one_patch = int(len(one_hot_vect) / nb_cut)
    patches = [one_hot_vect[i * len_one_patch:(i + 1) * len_one_patch] for i in range(nb_cut)]
    return patches


def accuracy_score(y_true, y_pred):
    '''
    Function that computes the accuracy score for our prediction
    Param: @y_true: true label
    @y_pred: prediction to evaluate
    Return: value of the accuracy score
    '''
    return max(np.sum(np.array(np.array(y_true) == np.array(y_pred), dtype=np.int)),
               np.sum(np.array(np.array(y_true) != np.array(y_pred), dtype=np.int))) / len(y_true)


def train_test_split(*arrays, test_size=0.5):
    '''
    Function that split arrays and list in two sets
    Param: *arrays: arrays to split
    test_size: (float) size of the test dataset (between 0 and 1)
    '''
    list_to_return = []
    shape_data = len(arrays[0])
    list_indice_shuffle = list(range(shape_data))
    random.shuffle(list_indice_shuffle)
    list_train, list_test = list_indice_shuffle[:int(len(list_indice_shuffle) * (1 - test_size))], list_indice_shuffle[
                                                                                                   int(len(
                                                                                                           list_indice_shuffle) * (
                                                                                                               1 - test_size)):]
    for array in arrays:
        if isinstance(array, list):
            list_to_return.extend([list(np.array(array)[list_train]), list(np.array(array)[list_test])])
        else:
            list_to_return.extend([(np.array(array)[list_train]), (np.array(array)[list_test])])
    return list_to_return
