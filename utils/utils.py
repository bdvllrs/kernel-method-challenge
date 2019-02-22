from datetime import datetime

import pandas as pd
import numpy as np
import random
import os
import kernels
import classifiers

__all__ = ['get_classifier', 'get_kernel', 'get_sets', 'save_submission', 'split_train_val']


def get_sets(path, slug="tr", merge=False, only=None):
    """
    Get data
    Args:
        path: path to data
        slug: tr for training and te for testing
        merge: if True, merge all three datafiles
        only: If given, between 0 and 2, only use the requested dataset.
    """
    path = os.path.abspath(os.path.join(os.curdir, path))
    datasets = []
    labels = []
    ids = []
    for k in range(3):
        data = pd.read_csv(os.path.join(path, "X{}{}.csv".format(slug, k)))
        datasets.append(data['seq'].values)
        ids.append(data['Id'])
        if slug == "tr":
            labels.append(pd.read_csv(os.path.join(path, "Y{}{}.csv".format(slug, k)))['Bound'].values)
    if merge:
        datasets = np.concatenate(datasets)
        ids = np.concatenate(ids)
        if slug == "tr":
            labels = np.concatenate(labels)
    elif only is not None:
        datasets = datasets[only]
        ids = ids[only]
        if slug == "tr":
            labels = labels[only]
    if slug == "tr":
        return datasets, labels
    return datasets, ids


def save_submission(conf, predictions, test_ids, accuracy):
    path = conf.submissions.path
    method = conf.classifiers.classifier
    kernel = conf.kernels.kernel
    ordered_pred = np.zeros_like(predictions)
    for k, idx in enumerate(test_ids):
        ordered_pred[idx] = predictions[k]
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    filename = "submission_{}_kernel_{}_val-acc_{}_{}".format(method, kernel, accuracy, date)
    path_csv = os.path.abspath(os.path.join(os.curdir, path, filename + ".csv"))
    path_yaml = os.path.abspath(os.path.join(os.curdir, path, filename + ".yaml"))
    with open(path_csv, 'w') as f:
        f.write('Id,Bound\n')
        for i in range(len(ordered_pred)):
            f.write(str(i)+','+str(ordered_pred[i])+'\n')
    conf.save(path_yaml)


def split_train_val(data, labels, ratio):
    keys = np.arange(data.shape[0])
    np.random.shuffle(keys)
    last_key = int(ratio * data.shape[0])
    train_data = data[keys[:last_key]]
    train_labels = labels[keys[:last_key]]
    val_data = data[keys[last_key:]]
    val_labels = labels[keys[last_key:]]
    return train_data, train_labels, val_data, val_labels


def get_kernel(kernel: str, kernel_type, gamma, degree, r, args) -> kernels.Kernel:
    kernel = kernel.lower()
    assert kernel in ['onehot', 'spectrum'], "Unknown requested kernel."

    if kernel == "spectrum":
        default_args = {"length": 3}
        default_args.update(args)
        kernel = kernels.SpectrumKernel(default_args['length'])
    else:
        kernel = kernels.OneHotKernel()
    kernel.set_args(kernel_type, gamma, degree, r)
    return kernel


def get_classifier(classifier: str, kernel, args) -> classifiers.Classifier:
    classifier = classifier.lower()
    assert classifier in ['svm', 'logistic-regression'], "Unknown requested classifier."

    if classifier == "svm":
        assert "C" in args.keys(), "`C` must be in config.classifiers.args for svm."
        assert "solver" in args.keys(), "`solver` must be in config.classifiers.args for svm."
        return classifiers.SVMClassifier(kernel, args['C'], args['solver'])
    elif classifier == "logistic-regression":
        assert "lambda" in args.keys(), "`lambda` must be in config.classifiers.args for logistic-regression."
        return classifiers.LogisticRegressionClassifier(kernel, args['lambda'])


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
