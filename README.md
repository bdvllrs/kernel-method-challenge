# Kernel Method Challenge

[Kaggle challenge](https://www.kaggle.com/c/kernel-methods-for-machine-learning-2018-2019) on DNA sequence classification for the MVA Msc Kernel Methods for Machine Learning Course.

## Memoizer
To improve performance, the class `utils.Memoizer` is used to save data to disk. The kernel have an instance in the `memoizer`
property. (Inside a kernel class `self.memoizer` accesses the data).

Some examples we suppose that we are in a kernel class and have access to `self.memoizer`
```python
self.memoizer["test.test2"] = [[1, 2, 3], [4, 5, 6]]  # save data in the storage
"test.test2" in self.memoizer  # test if "test.test2" key is in the memoizer
a = self.memoizer["test.test2"]  # Get values
```

The memoizer is saved onto the disk at the end of the `main.py` script.

Tip to save something depending on the data: use a hash!
```python
import hashlib

hash = hashlib.sha256(list_of_data.tostring()).hexdigest()
```

Examples in the default and Spectrum kernel.

## Kernels
Classes extending from `kernels.Kernel`.

### Methods to override
- `apply(self, embed1, embed2)` given two embedding vectors, returns `K(embed1, embed2)`. By default, use a liner kernel
(i.e. inner product).
- `embed(self, sequences)` takes a list of string sequence and returns a list of embedded vectors.

## Classifiers
Classes extending from `classifiers.Classifier`.

### Methods to override
- `fit(self, X, Y)` and computes and sets `self.alpha` which represents the parameters of the classifier
so that $f(x) = \sum_i \alpha_i K(x_i, x)$. Must also set `self.training_data = X` at the beginning for the `predict` method.

## Config
The file `config/default.yaml` is loaded, then all other files in the `config` folder are loaded
in alphabetical order, overriding (or adding) new values.

`defautl.yaml` is an example file and is the only versionned file.
To start, create a new file with personal values.

### kernels
Config of the different kernels.
`kernel` can be among :

#### onehot
Simple onehot where one letter is represented as a 4-dim onehot vector.

No args.

#### spectrum
Spectrum of the sequence, inspired from slide 55 of [http://members.cbio.mines-paristech.fr/~jvert/talks/060727mlss/mlss.pdf](http://members.cbio.mines-paristech.fr/~jvert/talks/060727mlss/mlss.pdf).

Args :
- length : Length of the words to do the spectrum on.


### classifiers
Config of the classifiers
Among :

#### svm

Args :
- C : C-svm.
- solver : solver to use. Among :
    - qp : quadratic program
    - sklearn : sklearn svm 
    
#### logistic-regression

Args:
- lambda: regularization constant.
