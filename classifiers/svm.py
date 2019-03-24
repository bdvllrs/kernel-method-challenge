__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
import cvxopt
from classifiers import Classifier


class SVMClassifier(Classifier):
    def __init__(self, kernel, C):
        super(SVMClassifier, self).__init__(kernel)
        self.C = C
        self.all_alphas = []

    def set_support_vectors(self):
        self.alpha = np.mean(self.all_alphas, axis=0)
        self.all_alphas = []
        # n = int(0.8 * len(self.alpha))
        # not_null = np.argsort(np.abs(self.alpha))[-n:]
        not_null = np.abs(self.alpha) > 1e-3
        self.alpha = self.alpha[not_null]
        print("{} support vectors found.".format(self.alpha.shape[0]))
        self.training_data = self.training_data[not_null]

    def fit(self, data, labels):
        """
        Solves the dual form using quadratic programming
        Solve $\min_x (1/2) x^T K x - y^Tx$
        subject to
        $Gx \le h$
        """
        self.training_data = data
        n = self.training_data.shape[0]
        K = self.kernel(data).astype(float) + np.eye(n) * 1e-6
        y = np.array(2 * labels - 1).astype(float)  # to {-1, 1}
        P = cvxopt.matrix(K)
        q = -cvxopt.matrix(y)
        G = cvxopt.matrix(np.concatenate([np.diag(y), -np.diag(y)], axis=0))
        h = cvxopt.matrix(np.concatenate([self.C * np.ones_like(y), np.zeros_like(y)]))
        self.alpha = np.array(cvxopt.solvers.qp(P, q, G, h)['x']).reshape(-1)
        self.all_alphas.append(self.alpha.copy())
