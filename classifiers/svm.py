__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
import cvxopt
from classifiers.default import Classifier


class SVMClassifier(Classifier):
    def __init__(self, kernel, C):
        super(SVMClassifier, self).__init__(kernel)
        self.C = C
        self.support_vectors = []

    def fit_dual(self, data, labels):
        """
        Solves the dual form using quadratic programming
        Solve $\min_x (1/2) x^T K x - y^Tx$
        subject to
        $Gx \le h$
        """
        n = data.shape[0]
        K = self.kernel(data).astype(float) + np.eye(n) * 1e-6
        P = cvxopt.matrix(K)
        y = np.array(2 * labels - 1).astype(float)  # to {-1, 1}
        q = -cvxopt.matrix(y)
        G = cvxopt.matrix(np.concatenate([np.diag(y), -np.diag(y)], axis=0))
        h = cvxopt.matrix(np.concatenate([self.C * np.ones_like(y), np.zeros_like(y)]))
        alpha = np.array(cvxopt.solvers.qp(P, q, G, h)['x']).reshape(-1)
        return alpha

    def fit_dual_with_intercept(self, data, labels):
        """
        Solvers the dual with intercept
        Args:
            data:
            labels:
        """
        n = data.shape[0]
        K = self.kernel(data).astype(float) + np.eye(n) * 1e-6
        y = np.array(2 * labels - 1).astype(float)  # to {-1, 1}
        ones = np.ones_like(y)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-ones)
        G = cvxopt.matrix(np.vstack([np.eye(n), -np.eye(n)]))
        h = cvxopt.matrix(np.hstack([self.C * ones, np.zeros_like(y)]))
        A = cvxopt.matrix(y, (1, n), "d")
        b = cvxopt.matrix(0.0)
        alpha = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).reshape(-1)
        return alpha

    def fit(self, data, labels):
        """
        Fit the SVM Classifier
        Args:
            data: X of shape [n, d] where n is the number of samples and d the dimension of the embedding space
            labels: labels
        Returns:
        """
        alpha = self.fit_dual_with_intercept(data, labels)
        self.alpha = alpha
        not_null = np.abs(self.alpha) > 1e-3
        self.alpha = self.alpha[not_null]
        print("{} support vectors found.".format(self.alpha.shape[0]))
        self.support_vectors = data[not_null]

