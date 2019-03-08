from numpy import mean
from utils import Config, get_classifier, get_sets, get_kernel, split_train_val, save_submission

config = Config('config')

kernel = get_kernel(config.kernels)

predictions = []
accuracies = []
ids = []

for idx in range(3):
    print(f"Dataset {idx + 1}.")
    train_set, train_labels = get_sets(config.data.path, "tr", idx=idx)
    test_data, test_ids = get_sets(config.data.path, "te", idx=idx)

    train_set = kernel.embed(train_set)

    test_data = kernel.embed(test_data)

    clf = get_classifier(config.classifiers.classifier, kernel, config.classifiers.args.values_())

    ratio = 1 - config.data.validation_set.ratio
    for i in range(config.data.bagging):
        print(f"Bagging step {i + 1}.")
        if ratio < 1:
            train_data, train_y, val_data, val_labels = split_train_val(train_set, train_labels, ratio=ratio)
        else:
            train_data, train_y = train_set, train_labels

        print("Fitting...")
        print(clf.fit(train_data, train_y))

        if ratio < 1:
            print("Evaluating...")
            results = clf.evaluate(val_data, val_labels)
            accuracies.append(results["Accuracy"])
            print(results)

    clf.set_support_vectors()

    print("Predicting...")
    predictions.extend(list(clf.predict(test_data)))
    ids.extend(list(test_ids))

kernel.memoizer.save()

if config.submissions.save:
    print("Saving submission...")
    print(predictions, ids)
    accuracy = mean(accuracies) if len(accuracies) > 0 else 0
    save_submission(config, predictions, ids, accuracy)
