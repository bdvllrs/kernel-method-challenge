from utils import Config, get_classifier, get_sets, get_kernel, kfold, split_train_val, save_submission

config = Config('config')

kernel = get_kernel(config.kernels.kernel, config.kernels.type, config.kernels.gamma, config.kernels.degree,
                    config.kernels.r, config.kernels.args.values_())

merge = not config.data.kfold and config.data.merge
only = None if config.data.kfold or config.data.merge else config.data.only

print("K-fold?", config.data.kfold)
print("Merge?", merge)
print("Only?", only)

train_data, train_labels = get_sets(config.data.path, "tr", merge=merge, only=only)
test_data, test_ids = get_sets(config.data.path, "te", merge=True)

if type(train_data) == list:
    for k in range(len(train_data)):
        train_data[k] = kernel.embed(train_data[k])
else:
    train_data = kernel.embed(train_data)

test_data = kernel.embed(test_data)

clf = get_classifier(config.classifiers.classifier, kernel, config.classifiers.args.values_())

if config.data.kfold:
    results, predictions = kfold(train_data, train_labels, test_data, clf)
    print("Best results.")
    print(results)
    print("Best prediction.")
    print(predictions)
else:
    ratio = 1 - config.data.validation_set.ratio
    train_data, train_labels, val_data, val_labels = split_train_val(train_data, train_labels, ratio=ratio)

    print("Fitting...")
    print(clf.fit(train_data, train_labels))

    print("Evaluating...")
    results = clf.evaluate(val_data, val_labels)
    print(results)

    print("Predicting...")
    predictions = clf.predict(test_data)
    print(predictions)

if config.submissions.save:
    print("Saving submission...")
    print(predictions, test_ids)
    save_submission(config, predictions, test_ids, results['Accuracy'])
