from copy import deepcopy
from numpy import mean
from utils.config import Config, update_config
from utils.gridsearch import GridSearch
from utils.utils import get_kernel, get_classifier, get_sets, save_submission, kfold, split_train_val
from kernels.default import SimpleMKL
from utils.pca import PCA

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

train_data, train_labels, val_data, val_labels = [], [], [], []

train_sets, train_set_labels = get_sets(glob_cfg.data.path, "tr")
test_data, test_ids = get_sets(glob_cfg.data.path, "te")

ratio = 1 - glob_cfg.data.validation_set.ratio

for idx in range(3):
    x1, y1, x2, y2 = split_train_val(train_sets[idx], train_set_labels[idx], ratio=ratio)
    train_data.append(x1)
    train_labels.append(y1)
    val_data.append(x2)
    val_labels.append(y2)

for set_idx in range(3):
    print(f"\nDataset {set_idx + 1}.")

    best_accuracy_set = 0
    best_predictions_set = None
    best_ids_set = None
    best_state_set = {}

    for hparam in gridsearch.hparams(set_idx):
        config = glob_cfg[f"set{set_idx}"]

        kernel = get_kernel(config.kernels)

        predictions = []
        accuracies = []
        ids = []

        train_x = kernel.embed(train_data[set_idx])
        train_y = train_labels[set_idx]
        val_x = kernel.embed(val_data[set_idx])
        val_y = val_labels[set_idx]
        test_x = kernel.embed(test_data[set_idx])

        clf = get_classifier(config.classifiers.classifier, kernel, config.classifiers.args.values_())

        if isinstance(kernel, SimpleMKL):
            kernel.fit(train_x, train_y, clf)

        pca = PCA(kernel)
        pca.apply(train_x, train_y)

        print("\nFitting...")
        clf.fit(train_x, train_y)

        print("\nEvaluating...")
        results = clf.evaluate(val_x, val_y)
        print("\n Val evaluation:", results)
        eval_train = clf.evaluate(train_x, train_y)
        print("\nEvaluation on train:", eval_train)

        accuracies.append(results["Accuracy"])

        print("\nPredicting...")
        predictions = list(clf.predict(test_x))
        ids = list(test_ids[set_idx])

        accuracy = mean(accuracies) if len(accuracies) > 0 else 0
        if accuracy > best_accuracy_set or best_predictions_set is None:
            best_accuracy_set, best_predictions_set = accuracy, predictions
            best_ids_set = ids
            best_state_set = hparam

        kernel.save()

    total_accuracy.append(best_accuracy_set)
    all_predictions.extend(best_predictions_set)
    all_ids.extend(best_ids_set)

    print(f"\n Best state found for set{set_idx} with values:")
    gridsearch.print_state(best_state_set, set_idx)

if glob_cfg.submissions.save:
    print("\nSaving submission for best model...")
    print(all_predictions)
    save_submission(glob_cfg, all_predictions, all_ids, total_accuracy)
