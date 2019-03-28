import numpy as np
from utils import metrics
from kernels import Kernel


class Classifier:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.alpha = None
        self.training_data = None
        self.support_vectors = None
        self.intercept = 0

    def set_support_vectors(self):
        pass

    def fit(self, data, labels):
        """
        Fit the data.
        """
        raise NotImplemented

    def reset(self):
        self.alpha = None
        self.training_data = None
        self.support_vectors = None

    def predict(self, data):
        """
        Args:
            data: DataFrame containing the data
        Returns: predicted labels for each row
        """
        assert self.support_vectors is not None, "Use fit before predicting."
        K = self.kernel(self.support_vectors, data)
        print(self.intercept)
        print(K)
        f = np.sign(K @ self.alpha + self.intercept)  # in {-1, 0, 1}
        return np.round((f + 1) / 2).astype(int)  # convert into {0, 1}

    def evaluate(self, data, labels):
        """
        Args:
            data: vector of embeddings
            labels: ground truth
        Returns: Metrics on the result.
        """
        predictions = self.predict(data)
        return {
            "Accuracy": metrics.accuracy(labels, predictions),
            "Recall": metrics.recall(labels, predictions),
            "Precision": metrics.precision(labels, predictions),
            "F1": metrics.f1(labels, predictions),
            "Predictions": predictions
        }
