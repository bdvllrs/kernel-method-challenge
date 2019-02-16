import numpy as np
from utils import metrics
from kernels import Kernel


class Classifier:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.alpha = None
        self.training_data = None

    def fit(self, data, labels):
        """
        Fit the data.
        """
        raise NotImplemented

    def predict(self, data):
        """
        Args:
            data: DataFrame containing the data
        Returns: predicted labels for each row
        """
        print("Predicting...")
        assert self.training_data is not None, "Use fit before predicting."
        K = self.kernel(self.training_data, data)
        f = np.sign(K @ self.alpha)  # in {-1, 0, 1}
        return np.round((f + 1) / 2)  # convert into {0, 1}

    def evaluate(self, data, labels):
        """
        Args:
            data: vector of embeddings
            labels: ground truth
        Returns: Metrics on the result.
        """
        print("Evaluating...")
        predictions = self.predict(data)
        return {
            "MSE": metrics.mse(labels, predictions),
            "Accuracy": metrics.accuracy(labels, predictions),
            # TODO: Add some other metrics like Recall, ...
        }
