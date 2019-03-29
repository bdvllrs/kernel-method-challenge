import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2D(x, y, labels):
    plt.figure()
    true_labels = np.where(labels == 1)
    false_labels = np.where(labels == 0)
    plt.plot(x[true_labels], y[true_labels], "rx")
    plt.plot(x[false_labels], y[false_labels], "bo")
    plt.show()


def plot_3D(x, y, z, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    true_labels = np.where(labels == 1)
    false_labels = np.where(labels == 0)
    ax.plot(x[true_labels], y[true_labels], z[true_labels], "rx")
    ax.plot(x[false_labels], y[false_labels], z[true_labels], "bo")
    plt.show()


class PCA:
    """
    Kernel PCA
    """

    def __init__(self, kernel, ndims=2):
        self.kernel = kernel
        self.ndims = ndims
        self.eig_vecs = None
        self.eig_vals = None

    def apply(self, train_data, train_labels):
        gram = self.kernel(train_data)
        n = gram.shape[0]
        eig_vals, eig_vecs = scipy.linalg.eigh(gram, eigvals=(n - self.ndims, n - 1))
        eig_vecs = eig_vecs / np.sqrt(eig_vals)
        self.eig_vecs = eig_vecs
        self.eig_vals = eig_vals

        projected_gram = gram @ eig_vecs.T

        if self.ndims == 2:
            plot_2D(projected_gram[:, 0], projected_gram[:, 1], train_labels)
        elif self.ndims == 3:
            plot_3D(projected_gram[:, 0], projected_gram[:, 1], projected_gram[:, 3], train_labels)


if __name__ == "__main__":
    pass
