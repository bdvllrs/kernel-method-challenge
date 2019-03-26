import numpy as np

__all__ = ['mse', 'accuracy']


def mse(truth, predicted):
    return np.sum((truth - predicted) ** 2)


def accuracy(truth, predicted):
    return np.sum(truth == predicted) / truth.shape[0]


def true_positive(truth, predicted):
    true_mask = truth == 1
    return len(predicted[true_mask] == 1)


def true_negative(truth, predicted):
    false_mask = truth == -1
    return len(predicted[false_mask] == -1)


def false_positive(truth, predicted):
    false_mask = truth == -1
    return len(predicted[false_mask] == 1)


def false_negative(truth, predicted):
    true_mask = truth == 1
    return len(predicted[true_mask] == -1)


def precision(truth, predicted):
    TP = true_positive(truth, predicted)
    FP = false_positive(truth, predicted)
    return TP / (TP + FP)


def recall(truth, predicted):
    TP = true_positive(truth, predicted)
    FN = false_negative(truth, predicted)
    return TP / (TP + FN)
