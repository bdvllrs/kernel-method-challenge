__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
import cvxopt
from classifiers.default import Classifier
import utils


class SVMClassifier(Classifier):
    def __init__(self, kernel, C):
        super(SVMClassifier, self).__init__(kernel)
        self.C = C
        self.all_alphas = []
        self.support_vectors = []

    def set_support_vectors(self):
        self.alpha = np.nanmean(self.all_alphas, axis=0)
        self.all_alphas = []
        # n = int(0.8 * len(self.alpha))
        # not_null = np.argsort(np.abs(self.alpha))[-n:]
        not_null = np.abs(self.alpha) > 1e-3
        self.alpha = self.alpha[not_null]
        print("{} support vectors found.".format(self.alpha.shape[0]))
        self.support_vectors = self.training_data[not_null]

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

    def fit(self, data, labels, split_ratio=1, bagging=1):
        """
        Fit the SVM Classifier
        Args:
            data: X of shape [n, d] where n is the number of samples and d the dimension of the embedding space
            labels: labels
            split_ratio: train split ratio
            bagging: number of bagging step
        Returns:
        """
        self.training_data = data
        for bagging_step in range(bagging):
            print(f"\nBagging step {bagging_step + 1}.")
            train_data, train_y, val_data, val_labels, train_mask, val_mask = utils.utils.split_train_val(data, labels, ratio=split_ratio)

            alpha = np.full_like(labels, np.nan, dtype=float)
            alpha[train_mask] = self.fit_dual(train_data, train_y)
            self.all_alphas.append(alpha)
        self.set_support_vectors()

        print("\nEvaluating...")
        results = self.evaluate(val_data, val_labels)
        print("\n Val evaluation:", results)
        eval_train = self.evaluate(train_data, train_y)
        print("\nEvaluation on train:", eval_train)

        return results
