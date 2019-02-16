import numpy as np
from qpsolvers import solve_qp
from classifiers import Classifier


class SVMClassifier(Classifier):
    def __init__(self, kernel, lbd, solver):
        super(SVMClassifier, self).__init__(kernel)
        self.lbd = lbd
        self.solver = solver

    def fit_quadratic_solver(self, data, labels):
        """
        Solve $\min_x (1/2) x^T K x - yx$
        subject to
        $Gx \le h$
        """
        self.training_data = data
        n = self.training_data.shape[0]
        K = self.kernel(data)
        K = K + np.eye(K.shape[0]) * 1e-5  # Otherwise not positive definite
        y = labels.astype(float)
        G = np.diag(y)
        G = np.concatenate([G, -G], axis=0)
        h = np.concatenate([1 / (2 * self.lbd * n) * np.ones(n), np.zeros(n)]).reshape((2 * n,))
        self.alpha = solve_qp(K, -y, G, h)
        return self.alpha

    def fit(self, data, labels):
        return self.fit_quadratic_solver(data, labels)
