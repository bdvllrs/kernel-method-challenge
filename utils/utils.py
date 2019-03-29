from datetime import datetime

import pandas as pd
import numpy as np
import os
import kernels
import classifiers

__all__ = ['get_classifier', 'get_kernel', 'get_sets', 'kfold', 'save_submission', 'split_train_val']


def get_sets(path, slug="tr"):
    """
    Get data
    Args:
        path: path to data
        slug: tr for training and te for testing
        idx: between 0 and 2, only use the requested dataset.
    """
    datasets = []
    all_labels = []
    all_ids = []
    for idx in range(3):
        path = os.path.abspath(os.path.join(os.curdir, path))
        data = pd.read_csv(os.path.join(path, "X{}{}.csv".format(slug, idx)))
        dataset = data['seq'].values
        ids = data['Id']
        if slug == "tr":
            labels = pd.read_csv(os.path.join(path, "Y{}{}.csv".format(slug, idx)))['Bound'].values
            datasets.append(dataset)
            all_labels.append(labels)
        else:
            datasets.append(dataset)
            all_ids.append(ids)
    if slug == "tr":
        return datasets, all_labels
    return datasets, all_ids


def save_submission(conf, predictions, test_ids, accuracy):
    path = conf.submissions.path
    ordered_pred = np.zeros_like(predictions)
    for k, idx in enumerate(test_ids):
        ordered_pred[idx] = predictions[k]
    date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    filename = f"submission_val-acc_{accuracy}_{date}"
    path_csv = os.path.abspath(os.path.join(os.curdir, path, filename + ".csv"))
    path_yaml = os.path.abspath(os.path.join(os.curdir, path, filename + ".yaml"))
    with open(path_csv, 'w') as f:
        f.write('Id,Bound\n')
        for i in range(len(ordered_pred)):
            f.write(str(i) + ',' + str(ordered_pred[i]) + '\n')
    conf.save_(path_yaml)


def split_train_val(data, labels, ratio):
    if ratio == 1:
        return data, labels, [], []
    keys = np.arange(data.shape[0])
    np.random.shuffle(keys)
    last_key = int(ratio * data.shape[0])
    train_data = data[keys[:last_key]]
    train_labels = labels[keys[:last_key]]
    val_data = data[keys[last_key:]]
    val_labels = labels[keys[last_key:]]
    return train_data, train_labels, val_data, val_labels


def get_kernel(conf) -> kernels.Kernel:
    all_kernels = conf.kernels.values_()
    list_kernels = []
    list_coefs = []
    for k in range(len(all_kernels)):
        kernel = all_kernels[k]["name"]
        if not conf.mkl:
            coef = conf.coefs[k]
            assert coef >= 0, f"Coefficient for kernel {k + 1} must be positive."
            list_coefs.append(coef)
        kernel_conf = conf.kernels[k]
        assert kernel in ['onehot', 'spectrum', "mismatch", "substring", "local-alignment"], "Unknown requested kernel."

        if kernel == "spectrum":
            default_args = {"length": 3}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.SpectrumKernel(conf.memoize, default_args['length'])
        elif kernel == "mismatch":
            default_args = {"k": 3, "m": 2}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.MismatchKernel(conf.memoize, default_args['k'], default_args['m'])
        elif kernel == "substring":
            default_args = {"length": 3, "lambda_decay": 0.05}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.SubstringKernel(conf.memoize, default_args['length'])
        elif kernel == "local-alignment":
            default_args = {"beta": 0.05, "d": 1, "e": 11}
            default_args.update(kernel_conf.args.values_())
            kernel = kernels.LocalAlignmentKernel(conf.memoize, default_args['beta'], default_args['d'],
                                                  default_args['e'])
        else:
            kernel = kernels.OneHotKernel(conf.memoize)
        kernel.set_args(kernel_conf.type, kernel_conf.gamma, kernel_conf.degree, kernel_conf.r)
        list_kernels.append(kernel)
    if conf.mkl:
        kernel = kernels.SimpleMKL(conf.memoize, list_kernels)
    else:
        kernel = kernels.SumKernel(conf.memoize, list_kernels, list_coefs)
    kernel.set_args(normalize=conf.normalize)
    return kernel


def get_classifier(classifier: str, kernel, args) -> classifiers.Classifier:
    classifier = classifier.lower()
    assert classifier in ['svm', 'logistic-regression'], "Unknown requested classifier."

    if classifier == "svm":
        assert "C" in args.keys(), "`C` must be in config.classifiers.args for svm."
        return classifiers.SVMClassifier(kernel, args['C'])
    elif classifier == "logistic-regression":
        assert "lambda" in args.keys(), "`lambda` must be in config.classifiers.args for logistic-regression."
        return classifiers.LogisticRegressionClassifier(kernel, args['lambda'])


def kfold(orig_data, orig_labels, test_data, clf: classifiers.Classifier):
    results = []
    predictions = []
    best_k = 0
    for k in range(len(orig_data)):
        print("Iteration", k + 1, "over", len(orig_data))
        data, labels = orig_data[:], orig_labels[:]
        val_data = data.pop(k)
        val_labels = labels.pop(k)
        train_data = np.concatenate(data)
        train_labels = np.concatenate(labels)
        print("Fitting...")
        clf.fit(train_data, train_labels)
        print("Evaluating...")
        results.append(clf.evaluate(val_data, val_labels))
        print(results[k])
        print("Predicting...")
        predictions.append(clf.predict(test_data))
        print(predictions[k])
        if results[k]["Accuracy"] > results[best_k]["Accuracy"]:
            best_k = k
    return results[best_k], predictions[best_k]
