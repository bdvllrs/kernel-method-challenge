__author__ = "Benjamin Devillers (bdvllrs)"

import numpy as np
import cvxopt
from classifiers.default import Classifier


class SVMClassifier(Classifier):
    def __init__(self, kernel, C):
        super(SVMClassifier, self).__init__(kernel)
        self.C = C
        self.support_vectors = []
        self.labels = None

    def fit_dual(self, K, y):
        """
        Solves the dual form using quadratic programming
        Solve $\min_x (1/2) x^T K x - y^Tx$
        subject to
        $Gx \le h$

        Args:
            K: gram
            y: in {0, 1}
        """
        P = cvxopt.matrix(K)
        q = -cvxopt.matrix(y)
        G = cvxopt.matrix(np.concatenate([np.diag(y), -np.diag(y)], axis=0))
        h = cvxopt.matrix(np.concatenate([self.C * np.ones_like(y), np.zeros_like(y)]))
        alpha = np.array(cvxopt.solvers.qp(P, q, G, h)['x']).reshape(-1)
        return alpha

    def fit_dual_with_intercept(self, K, y):
        """
        Solvers the dual with intercept
        Args:
            K:
            y:
        """
        n = y.shape[0]
        P = cvxopt.matrix(np.diag(y) @ K @ np.diag(y) + np.eye(n) * 1e-6)
        q = cvxopt.matrix(-np.ones(n))
        if not self.C:
            G = cvxopt.matrix(-np.eye(n))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.vstack([np.eye(n), -np.eye(n)]))
            h = cvxopt.matrix(np.hstack([self.C * np.ones(n), np.zeros(n)]))
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
        n = data.shape[0]
        labels = np.array(2 * labels - 1).astype(float)  # to {-1, 1}
        K = self.kernel(data).astype(float) + np.eye(n) * 1e-6

        alpha = self.fit_dual_with_intercept(K, labels)
        self.alpha = alpha
        support = self.alpha > 1e-4
        self.alpha = self.alpha[support] * labels[support]
        print("{} support vectors found.".format(self.alpha.shape[0]))
        self.support_vectors = data[support]
        K = K[support][:, support]
        alpha = self.alpha.reshape(-1, 1)
        w = np.sum(alpha * K, axis=0)
        true_labels = np.where(labels[support] == 1)
        false_labels = np.where(labels[support] == -1)
        self.intercept = -(np.max(w[false_labels]) + np.min(w[true_labels])) / 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import kernels
    from utils.config import Config

    x1 = np.random.uniform(0, 1, size=(100, 1))
    y1 = np.random.uniform(0, 0.5, size=(100, 1))
    x2 = np.random.uniform(0, 1, size=(200, 1))
    y2 = np.random.uniform(0.7, 2, size=(200, 1))
    set1 = np.concatenate([x1, y1], axis=1)
    set2 = np.concatenate([x2, y2], axis=1)
    dataset = np.vstack((set1, set2))
    plt.plot(x1, y1, "xr")
    plt.plot(x2, y2, "ob")
    plt.show()
    labels = np.array([0] * 100 + [1] * 200)
    idx = np.arange(0, len(labels))
    np.random.shuffle(idx)
    train_set, train_labels = dataset[idx], labels[idx]

    config = Config("../config/")
    kernel = kernels.Kernel(config["global"].kernels.memoize)
    clf = SVMClassifier(kernel, C=1)

    clf.fit(train_set, train_labels)

    w1 = np.sum(clf.alpha * clf.support_vectors[:, 0])
    w2 = np.sum(clf.alpha * clf.support_vectors[:, 1])
    # b = w1 *
    t = np.linspace(0, 1, 3)
    line = (-clf.intercept - t * w1) / w2
    x, y = train_set[:, 0], train_set[:, 1]
    mask_1 = train_labels == 0
    mask_2 = train_labels == 1
    x1, y1 = x[mask_1], y[mask_1]
    x2, y2 = x[mask_2], y[mask_2]
    plt.plot(x1, y1, "xg")
    plt.plot(x2, y2, "xr")
    plt.plot(t, line, "-r")
    plt.plot(t, line + 1 / w2, "-r")
    plt.plot(t, line - 1 / w2, "-r")
    plt.show()
