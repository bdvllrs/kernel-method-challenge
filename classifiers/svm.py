import numpy as np
import cvxopt
from sklearn import svm  # Not used in the challenge, only for testing purposes.
from classifiers import Classifier


class SVMClassifier(Classifier):
    def __init__(self, kernel, C, solver):
        super(SVMClassifier, self).__init__(kernel)
        self.C = C
        self.solver = solver
        self.sklearn_clf = None

    def predict(self, data):
        if self.solver == "sklearn" and self.sklearn_clf is not None:
            K = self.kernel(self.training_data, data)
            return self.sklearn_clf.predict(K)
        return super(SVMClassifier, self).predict(data)

    def fit_quadratic_solver(self, data, labels):
        """
        Solves the dual form using quadratic programming
        Solve $\min_x (1/2) x^T K x - y^Tx$
        subject to
        $Gx \le h$
        """
        n = self.training_data.shape[0]
        K = self.kernel(data).astype(float) + np.eye(n) * 1e-6
        y = np.array(2 * labels - 1).astype(float)  # to {-1, 1}
        P = cvxopt.matrix(K)
        q = -cvxopt.matrix(y)
        G = cvxopt.matrix(np.concatenate([np.diag(y), -np.diag(y)], axis=0))
        h = cvxopt.matrix(np.concatenate([self.C * np.ones_like(y), np.zeros_like(y)]))
        self.alpha = np.array(cvxopt.solvers.qp(P, q, G, h)['x']).reshape(-1)
        # Only support vectors
        not_null = np.abs(self.alpha) > 1e-4
        self.alpha = self.alpha[not_null]
        print("{} support vectors found.".format(self.alpha.shape[0]))
        self.training_data = self.training_data[not_null]

    def fit_sklearn(self, data, labels):
        """
        For testing. Not used for the challenge.
        """
        self.sklearn_clf = svm.SVC(C=self.C, kernel="precomputed", gamma="scale")
        K = self.kernel(data)
        self.sklearn_clf.fit(K, labels)

    def fit(self, data, labels):
        self.training_data = data
        print("Fitting using {} solver...".format(self.solver))
        if self.solver == "sklearn":
            return self.fit_sklearn(data, labels)
        return self.fit_quadratic_solver(data, labels)
