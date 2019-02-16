import numpy as np
from utils import metrics
from kernels import Kernel


class Classifier:
    def __init__(self, training_data, kernel: Kernel):
        self.training_data = training_data
        self.kernel = kernel
        print('Computing Gram matrix...')
        self.train_K = kernel(training_data)
        print('Computed.')
        self.alpha = np.zeros(self.training_data.shape[0])

    def fit(self):
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
        K = self.kernel(self.training_data.values, data)
        f = np.sign(K @ self.alpha)  # in {-1, 0, 1}
        return np.round((f + 1) / 2)  # convert into {0, 1}

    def evaluate(self, data):
        """
        Args:
            data: DataFrame containing data and labels
        Returns: Metrics on the result.
        """
        predictions = self.predict(data)
        return {
            "MSE": metrics.mse(data['Bound'], predictions),
            "Accuracy": metrics.accuracy(data['Bound'], predictions),
            # TODO: Add some other metrics like Recall, ...
        }
