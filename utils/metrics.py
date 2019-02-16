import numpy as np

__all__ = ['mse', 'accuracy']


def mse(truth, predicted):
    return np.sum((truth - predicted) ** 2)


def accuracy(truth, predicted):
    return np.sum(truth == predicted) / truth.shape[0]
