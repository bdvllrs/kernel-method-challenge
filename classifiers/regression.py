__author__ = "Benjamin Devillers (bdvllrs)"


import numpy as np
from classifiers import Classifier
import scipy.optimize as optim

__all__ = ["LogisticRegressionClassifier"]


def sigmoid(u):
    return np.reciprocal(1 + np.exp(-u))


def logistic_loss(u):
    return np.log(1 + np.exp(-u))


def grad_logistic_loss(u):
    return -sigmoid(-u)


def hess_logistic_loss(u):
    return sigmoid(u) * sigmoid(-u)


class LogisticRegressionClassifier(Classifier):
    def __init__(self, kernel, lbd):
        super(LogisticRegressionClassifier, self).__init__(kernel)
        self.lbd = lbd

    def fit(self, data, labels):
        self.training_data = data
        n = data.shape[0]
        self.alpha = np.zeros(n, dtype=float)
        K = self.kernel(data).astype(float)
        y = np.array(2 * labels - 1, dtype=float)  # to {-1, 1}
        lbd = self.lbd

        def objective_fn(u):
            return np.sum(logistic_loss(y * (K @ u))) / n + lbd * u.T @ K @ u / 2

        def grad_fn(u):
            P = np.diag(grad_logistic_loss(y * (K @ u)))
            return K @ P @ y / n + lbd * K @ u

        def hess_fn(u):
            W = np.diag(hess_logistic_loss(y * (K @ u)))
            return K @ W @ K + lbd * K

        self.alpha = optim.minimize(objective_fn, self.alpha, method="Newton-CG", jac=grad_fn, hess=hess_fn)['x']
