import numpy as np
from qpsolvers import solve_qp
from sklearn import svm  # Not used in the challenge, only for testing purposes.
from classifiers import Classifier


class SVMClassifier(Classifier):
    def __init__(self, kernel, lbd, solver):
        super(SVMClassifier, self).__init__(kernel)
        self.lbd = lbd
        self.solver = solver
        self.sklearn_clf = None

    def predict(self, data):
        if self.solver == "sklearn" and self.sklearn_clf is not None:
            K = self.kernel(self.training_data, data)
            return self.sklearn_clf.predict(K)
        return super(SVMClassifier, self).predict(data)

    def fit_quadratic_solver(self, data, labels):
        """
        Solve $\min_x (1/2) x^T K x - yx$
        subject to
        $Gx \le h$
        """
        n = self.training_data.shape[0]
        K = self.kernel(data)
        K = K + np.eye(K.shape[0]) * 1e-5  # Otherwise not positive definite
        y = labels.astype(float)
        G = np.diag(y)
        G = np.concatenate([G, -G], axis=0)
        h = np.concatenate([1 / (2 * self.lbd * n) * np.ones(n), np.zeros(n)]).reshape((2 * n,))
        self.alpha = solve_qp(K, -y, G, h)
        return self.alpha

    def fit_sklearn(self, data, labels):
        """
        For testing. Not used for the challenge.
        """
        self.sklearn_clf = svm.SVC(kernel="precomputed", gamma="scale")
        K = self.kernel(data)
        self.sklearn_clf.fit(K, labels)

    def fit(self, data, labels):
        self.training_data = data
        print("Fitting using {} solver...".format(self.solver))
        if self.solver == "sklearn":
            return self.fit_sklearn(data, labels)
        return self.fit_quadratic_solver(data, labels)
