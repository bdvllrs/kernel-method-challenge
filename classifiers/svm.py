import numpy as np
from classifiers import Classifier


class SVMClassifier(Classifier):
    def __init__(self, training_data, kernel, lbd):
        super(SVMClassifier, self).__init__(training_data, kernel)
        self.lbd = lbd

    def fit(self):
        def hinge(x):
            return np.max(1 - x, 0)

        # TODO: subgradient descent
        return self.alpha
