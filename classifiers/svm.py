import numpy as np
from qpsolvers import solve_qp
from classifiers import Classifier


class SVMClassifier(Classifier):
    def __init__(self, training_data, kernel, lbd):
        super(SVMClassifier, self).__init__(training_data, kernel)
        self.lbd = lbd

    def fit(self):
        """
        Solve $\min_x (1/2) x^T K x - yx$
        subject to
        $Gx \le h$
        """
        n = self.training_data.shape[0]
        K = self.train_K
        K = K + np.eye(K.shape[0]) * 1e-5  # Otherwise not positive definite
        y = self.training_data['Bound'].values.astype(float)
        G = np.diag(y)
        G = np.concatenate([G, -G], axis=0)
        h = np.concatenate([1 / (2 * self.lbd * n) * np.ones(n), np.zeros(n)]).reshape((2 * n,))
        self.alpha = solve_qp(K, -y, G, h)
        return self.alpha
