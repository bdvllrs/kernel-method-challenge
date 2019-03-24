from datetime import datetime

import pandas as pd
import numpy as np
import random
import os
import kernels
import classifiers

__all__ = ['get_classifier', 'get_kernel', 'get_set', 'kfold', 'save_submission', 'split_train_val']


def get_set(path, slug="tr", idx=0):
    """
    Get data
    Args:
        path: path to data
        slug: tr for training and te for testing
        idx: between 0 and 2, only use the requested dataset.
    """
    path = os.path.abspath(os.path.join(os.curdir, path))
    data = pd.read_csv(os.path.join(path, "X{}{}.csv".format(slug, idx)))
    dataset = data['seq'].values
    ids = data['Id']
    if slug == "tr":
        labels = pd.read_csv(os.path.join(path, "Y{}{}.csv".format(slug, idx)))['Bound'].values
        return dataset, labels
    return dataset, ids


def save_submission(conf, predictions, test_ids, accuracy):
    path = conf.submissions.path
    ordered_pred = np.zeros_like(predictions)
    for k, idx in enumerate(test_ids):
        ordered_pred[idx] = predictions[k]
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    filename = f"submission_val-acc_{accuracy}_{date}"
    path_csv = os.path.abspath(os.path.join(os.curdir, path, filename + ".csv"))
    path_yaml = os.path.abspath(os.path.join(os.curdir, path, filename + ".yaml"))
    with open(path_csv, 'w') as f:
        f.write('Id,Bound\n')
        for i in range(len(ordered_pred)):
            f.write(str(i) + ',' + str(ordered_pred[i]) + '\n')
    conf.save_(path_yaml)


def split_train_val(data, labels, ratio):
    keys = np.arange(data.shape[0])
    np.random.shuffle(keys)
    last_key = int(ratio * data.shape[0])
    train_data = data[keys[:last_key]]
    train_labels = labels[keys[:last_key]]
    val_data = data[keys[last_key:]]
    val_labels = labels[keys[last_key:]]
    return train_data, train_labels, val_data, val_labels


def get_kernel(conf) -> kernels.Kernel:
    all_kernels = conf.kernels.values_()
    list_kernels = []
    list_coefs = []
    for k in range(len(all_kernels)):
        kernel = all_kernels[k]["name"]
        coef = conf.coefs[k]
        assert coef >= 0, f"Coefficient for kernel {k + 1} must be positive."
        list_coefs.append(coef)
        kernel_conf = conf.kernels[k]
        assert kernel in ['onehot', 'spectrum', "mismatch", "substring", "local-alignment"], "Unknown requested kernel."

        if kernel == "spectrum":
            default_args = {"length": 3}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.SpectrumKernel(conf.memoize, default_args['length'])
        elif kernel == "mismatch":
            default_args = {"length": 3}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.MismatchKernel(conf.memoize, default_args['length'])
        elif kernel == "substring":
            default_args = {"length": 3, "lambda_decay": 0.05}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.SubstringKernel(conf.memoize, default_args['length'])
        elif kernel == "local-alignment":
            default_args = {"beta": 0.05, "d": 1, "e": 11}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.LocalAlignmentKernel(conf.memoize, default_args['beta'], default_args['d'],
                                                  default_args['e'])
        else:
            kernel = kernels.OneHotKernel(conf.memoize)
        kernel.set_args(kernel_conf.type, kernel_conf.gamma, kernel_conf.degree, kernel_conf.r)
        list_kernels.append(kernel)
    kernel = kernels.SumKernel(conf.memoize, list_kernels, list_coefs)
    return kernel


def get_classifier(classifier: str, kernel, args) -> classifiers.Classifier:
    classifier = classifier.lower()
    assert classifier in ['svm', 'logistic-regression'], "Unknown requested classifier."

    if classifier == "svm":
        assert "C" in args.keys(), "`C` must be in config.classifiers.args for svm."
        return classifiers.SVMClassifier(kernel, args['C'])
    elif classifier == "logistic-regression":
        assert "lambda" in args.keys(), "`lambda` must be in config.classifiers.args for logistic-regression."
        return classifiers.LogisticRegressionClassifier(kernel, args['lambda'])


def kfold(orig_data, orig_labels, test_data, clf: classifiers.Classifier):
    results = []
    predictions = []
    best_k = 0
    for k in range(len(orig_data)):
        print("Iteration", k + 1, "over", len(orig_data))
        data, labels = orig_data[:], orig_labels[:]
        val_data = data.pop(k)
        val_labels = labels.pop(k)
        train_data = np.concatenate(data)
        train_labels = np.concatenate(labels)
        print("Fitting...")
        clf.fit(train_data, train_labels)
        print("Evaluating...")
        results.append(clf.evaluate(val_data, val_labels))
        print(results[k])
        print("Predicting...")
        predictions.append(clf.predict(test_data))
        print(predictions[k])
        if results[k]["Accuracy"] > results[best_k]["Accuracy"]:
            best_k = k
    return results[best_k], predictions[best_k]


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
