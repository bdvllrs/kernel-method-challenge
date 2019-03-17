from copy import deepcopy
from numpy import mean
from utils import Config, get_classifier, get_sets, get_kernel, split_train_val, save_submission, GridSearch
from utils.config import update_config

glob_cfg = Config('config')

# Start by updating the glob_cfg set0, set1 and set2 with global glob_cfg
# Deepcopy is used because update_config has some side effects...
glob_cfg.set_("set0", update_config(deepcopy(glob_cfg["global"].values_()), glob_cfg.set0.values_()))
glob_cfg.set_("set1", update_config(deepcopy(glob_cfg["global"].values_().copy()), glob_cfg.set1.values_()))
glob_cfg.set_("set2", update_config(deepcopy(glob_cfg["global"].values_().copy()), glob_cfg.set2.values_()))

gridsearch = GridSearch(glob_cfg)

total_accuracy = []
all_predictions = []
all_ids = []

for idx in range(3):
    print(f"\nDataset {idx + 1}.")

    best_accuracy_set = 0
    best_predictions_set = None
    best_ids_set = None
    best_state_set = {}

    for state in gridsearch.states(idx):
        config = glob_cfg[f"set{idx}"]

        kernel = get_kernel(config.kernels)

        predictions = []
        accuracies = []
        ids = []

        train_set, train_labels = get_sets(glob_cfg.data.path, "tr", idx=idx)
        test_data, test_ids = get_sets(glob_cfg.data.path, "te", idx=idx)

        train_set = kernel.embed(train_set)

        test_data = kernel.embed(test_data)

        clf = get_classifier(config.classifiers.classifier, kernel, config.classifiers.args.values_())

        ratio = 1 - glob_cfg.data.validation_set.ratio
        for i in range(glob_cfg.data.bagging):
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
        predictions = list(clf.predict(test_data))
        ids = list(test_ids)

        accuracy = mean(accuracies) if len(accuracies) > 0 else 0
        if accuracy > best_accuracy_set or best_predictions_set is None:
            best_accuracy_set, best_predictions_set = accuracy, predictions
            best_ids_set = ids
            best_state_set = state

        kernel.memoizer.save()

    total_accuracy.append(best_accuracy_set)
    all_predictions.extend(best_predictions_set)
    all_ids.extend(best_ids_set)

    print(f"\n Best state found for set{idx} with values:")
    gridsearch.print_state(best_state_set, idx)


if glob_cfg.submissions.save:
    print("\nSaving submission for best model...")
    print(all_predictions)
    save_submission(glob_cfg, all_predictions, all_ids, total_accuracy)
