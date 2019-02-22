# Kernel_method_challenge

Kaggle challenge on DNA sequence classification for Kernel Methods for Machine Learning Course.

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