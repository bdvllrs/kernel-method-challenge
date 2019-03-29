import numpy as np
import hashlib
from utils.memoizer import Memoizer
from tqdm import tqdm
from utils.projected_gradient import projected_gradient
#from classifiers.svm import SVMClassifier

__all__ = ['Kernel', 'OneHotKernel', 'SumKernel', 'SimpleMKL']


class Kernel:
    def __init__(self, memoize_conf):
        self.memoizer = Memoizer(memoize_conf, type(self).__name__)
        self.type = "linear"
        self.gamma = "auto"
        self.d = 2
        self.r = 0
        self.normalize = False
        self.last_gram = None

    def config_digest(self):
        """
        To find which value to get in the memoizer.
        2 diff configs should yield 2 different results in the memoizer.
        """
        return ""

    def save(self):
        self.memoizer.save()

    def set_args(self, kernel_type="linear", gamma="auto", degree=2, r=0, normalize=False):
        """
        Args:
            normalize: if True, normalizes the gram matrix: K_norm(x, y) = K(x, y) / sqrt(K(x, x)K(y, y))
            kernel_type:  in (linear, polynomial, gaussian, sigmoid)
            gamma: if None, auto defined
            degree: if polynomial kernel, degree of the polynomial. Default 2.
            r: added constant for polynomial and sigmoid kernel. Default 0.
        """
        self.type = kernel_type
        self.gamma = gamma
        self.d = degree
        self.r = r
        self.normalize = normalize

    def __call__(self, data_1, data_2=None):
        """
        Returns the Gram Matrix using the embed values of the self.embed function
        Args:
            data_1: array of embeddings
            data_2: (optional) array of embeddings 2. Default: data_2 = data_1.
        """
        is_same_data = data_2 is None
        if data_2 is None:
            data_2 = data_1
        # Get hashes for memoization
        hash_1 = hashlib.sha256(data_1.tostring()).hexdigest() + self.config_digest()
        hash_2 = hashlib.sha256(data_2.tostring()).hexdigest() + self.config_digest()
        if "gram.{}.{}".format(hash_1, hash_2) in self.memoizer:
            print('Using memoized data.')
            path = "gram.{}.{}".format(hash_1, hash_2)
            return self.memoizer[path]
        if "gram.{}.{}".format(hash_2, hash_1) in self.memoizer:
            print('Using memoized data.')
            path = "gram.{}.{}".format(hash_2, hash_1)
            return self.memoizer[path]
        gram = np.ones((len(data_2), len(data_1))) * -1
        print("Computing gram...")
        with tqdm(total=len(data_1) * len(data_2)) as progress_bar:
            for j, y in enumerate(data_2):
                for i, x in enumerate(data_1):
                    if not is_same_data or j <= i:
                        gram[j, i] = self.apply(x, y, i, j)
                        progress_bar.update(1)
                    if is_same_data and j <= i:
                        gram[i, j] = gram[j, i]
                        progress_bar.update(1)
        self.memoizer["gram.{}.{}".format(hash_1, hash_2)] = gram
        if self.normalize:
            diag = np.diag(gram)
            norm_consts = np.sqrt(np.outer(diag, diag))
            gram = gram / norm_consts
        self.last_gram = gram
        return gram

    def apply(self, embed1, embed2, idx1, idx2):
        """
        Returns the value of K(embed1, embed2)
        """
        if self.gamma == "auto":
            self.gamma = 1 / embed1.shape[0]
        if self.type == "polynomial":
            return self._polynomial_kernel(embed1, embed2)
        elif self.type == "gaussian":
            return self._gaussian_kernel(embed1, embed2)
        elif self.type == "sigmoid":
            return self._sigmoid_kernel(embed1, embed2)
        return self._linear_kernel(embed1, embed2)

    def _linear_kernel(self, embed1, embed2):
        return np.inner(embed1, embed2)

    def _polynomial_kernel(self, embed1, embed2):
        return (self.gamma * self._linear_kernel(embed1, embed2) + self.r) ** self.d

    def _gaussian_kernel(self, embed1, embed2):
        return np.exp(-self.gamma * np.linalg.norm(embed1 - embed2) ** 2)

    def _sigmoid_kernel(self, embed1, embed2):
        return np.tanh(self.gamma * self._linear_kernel(embed1, embed2) + self.r)

    def embed(self, sequences):
        # Add the hash so that it is data dependant.
        # path = "embeddings." + hashlib.sha256(sequences.tostring()).hexdigest() + self.config_digest()
        # if path in self.memoizer:
        #     return self.memoizer[path]
        embeddings = np.array([self.embed_one(sequences[k]) for k in range(sequences.shape[0])])
        # self.memoizer[path] = embeddings
        return embeddings

    def embed_one(self, sequence):
        raise NotImplementedError


class OneHotKernel(Kernel):
    """
    Returns a onehot vector.
    As an example... bad in practise.
    """

    @staticmethod
    def to_onehot(sequence):
        letters = {"T": 0, "G": 1, "A": 2, "C": 3}
        onehot = np.zeros((len(sequence), 4))
        sequence = np.array([letters[letter] for letter in sequence])
        onehot[np.arange(sequence.shape[0]), sequence] = 1
        return np.ravel(onehot)

    def embed(self, sequences):
        return np.array([OneHotKernel.to_onehot(sequences[k]) for k in range(sequences.shape[0])])


class SumKernel(Kernel):
    def __init__(self, memoize_conf, kernels, coefs):
        """
        Args:
            memoize_conf:  conf for memoization
            kernels: list of all instaciated kernels.
            coefs: Coeficients to combine the kernels
        """
        super(SumKernel, self).__init__(memoize_conf)
        self.kernels = kernels
        self.coefs = coefs

    def config_digest(self):
        return "-".join(type(kernel).__name__ + kernel.config_digest() for kernel in self.kernels)

    def save(self):
        for kernel in self.kernels:
            kernel.save()
        self.memoizer.save()

    def embed(self, sequences):
        embeds = []
        for kernel in self.kernels:
            embeds.append(kernel.embed(sequences))
        embeddinds = [[embeds[k][i] for k in range(len(self.kernels))] for i in range(len(sequences))]
        return np.array(embeddinds)

    def apply(self, embed1, embed2, idx1, idx2):
        result = 0
        for k in range(len(embed1)):
            result += self.coefs[k] * self.kernels[k].apply(embed1[k], embed2[k], idx1, idx2)
        return result
    
class SimpleMKL(Kernel):
    def __init__(self, memoize_conf, kernels):
        """
        Args:
            memoize_conf:  conf for memoization
            kernels: list of all instanciated kernels.
            coefs: Coeficients to combine the kernels
        """
        super(SimpleMKL, self).__init__(memoize_conf)
        self.kernels = kernels
        self.coefs = None

    def config_digest(self):
        return "-".join(type(kernel).__name__ + kernel.config_digest() for kernel in self.kernels)

    def save(self):
        for kernel in self.kernels:
            kernel.save()
        self.memoizer.save()
        
        
    def fit(self, trainset, labels, clf, tol=1e-2, n_iter=100):
        M = len(self.kernels)
        self.coefs = np.ones((M,))/M
        i = 0
        lbd = 1/(2*len(trainset)*clf.C)
        projection_fct = lambda u: np.abs(u) / np.sum(np.abs(u)) 
        old_coefs = self.coefs
        converged = False
        kernel_grams = [self.kernels[i](trainset[:, i]).astype(float) for i in range(M)]
        while not converged and i < n_iter:
            K = np.sum([self.coefs[i]*kernel_grams[i] for i in range(M)], axis=0)
            clf.fit(trainset, labels, gram=K)
            alpha = clf.alpha
            grad_J = lambda x: np.array([-lbd * alpha.T @ self.kernels[i].last_gram[clf.support_idx,:][:,clf.support_idx] @ alpha for i in range(M)])
            print(grad_J(0).shape)
            print(projection_fct(self.coefs))
            self.coefs = projected_gradient(self.coefs, grad_J, projection_fct)
            if np.linalg.norm(old_coefs - self.coefs) < tol:
                converged = True
                break
            old_coefs = self.coefs
        print('Converged : coefs = {}'.format(self.coefs))

                
    def embed(self, sequences):
        embeds = []
        for kernel in self.kernels:
            embeds.append(kernel.embed(sequences))
        embeddinds = [[embeds[k][i] for k in range(len(self.kernels))] for i in range(len(sequences))]
        return np.array(embeddinds)

    def apply(self, embed1, embed2, idx1, idx2):
        result = 0
        for k in range(len(embed1)):
            result += self.coefs[k] * self.kernels[k].apply(embed1[k], embed2[k], idx1, idx2)
        return result
    
    

