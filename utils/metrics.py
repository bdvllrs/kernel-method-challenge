import numpy as np

__all__ = ['mse', 'accuracy']


def mse(truth, predicted):
    return np.sum((truth - predicted) ** 2)


def accuracy(truth, predicted):
    return np.sum(truth == predicted) / truth.shape[0]


def true_positive(truth, predicted):
    return np.sum(np.logical_and(predicted == 1, truth == 1))


def true_negative(truth, predicted):
    return np.sum(np.logical_and(predicted == 0, truth == 0))


def false_positive(truth, predicted):
    return np.sum(np.logical_and(predicted == 1, truth == 0))


def false_negative(truth, predicted):
    return np.sum(np.logical_and(predicted == 0, truth == 1))


def precision(truth, predicted):
    TP = true_positive(truth, predicted)
    FP = false_positive(truth, predicted)
    return TP / (TP + FP)


def recall(truth, predicted):
    TP = true_positive(truth, predicted)
    FN = false_negative(truth, predicted)
    return TP / (TP + FN)


def f1(truth, predicted):
    TP = true_positive(truth, predicted)
    FN = false_negative(truth, predicted)
    FP = false_positive(truth, predicted)
    return 2 * TP / (2 * TP + FP + FN)
