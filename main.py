from numpy import mean
from utils import Config, get_classifier, get_sets, get_kernel, split_train_val, save_submission, GridSearch

config = Config('config')

gridsearch = GridSearch(config)

best_accuracy = 0
best_predictions = None
best_ids = None
best_state = {}

for state in gridsearch.states():

    kernel = get_kernel(config.kernels)

    predictions = []
    accuracies = []
    ids = []

    for idx in range(3):
        print(f"\nDataset {idx + 1}.")
        train_set, train_labels = get_sets(config.data.path, "tr", idx=idx)
        test_data, test_ids = get_sets(config.data.path, "te", idx=idx)

        train_set = kernel.embed(train_set)

        test_data = kernel.embed(test_data)

        clf = get_classifier(config.classifiers.classifier, kernel, config.classifiers.args.values_())

        ratio = 1 - config.data.validation_set.ratio
        for i in range(config.data.bagging):
            print(f"\nBagging step {i + 1}.")
            if ratio < 1:
                train_data, train_y, val_data, val_labels = split_train_val(train_set, train_labels, ratio=ratio)
            else:
                train_data, train_y = train_set, train_labels

            print("\nFitting...")
            print(clf.fit(train_data, train_y))

            if ratio < 1:
                print("\nEvaluating...")
                results = clf.evaluate(val_data, val_labels)
                print(clf.predict(val_data))
                accuracies.append(results["Accuracy"])
                print("\n Val evaluation:", results)

        clf.set_support_vectors()
        eval_train = clf.evaluate(train_data, train_y)
        print("\nEvaluation on train:", eval_train)

        print("\nPredicting...")
        predictions.extend(list(clf.predict(test_data)))
        ids.extend(list(test_ids))

    kernel.memoizer.save()

    accuracy = mean(accuracies) if len(accuracies) > 0 else 0
    if accuracy > best_accuracy or best_predictions is None:
        best_accuracy, best_predictions = accuracy, predictions
        best_ids = ids
        best_state = state

print("\n Best state found with values:")
gridsearch.print_state(best_state)

if config.submissions.save:
    print("\nSaving submission for best model...")
    print(best_predictions)
    save_submission(config, best_predictions, best_ids, best_accuracy)
