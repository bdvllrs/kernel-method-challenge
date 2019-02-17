from utils import Config, get_classifier, get_sets, get_kernel, split_train_val, save_submission

config = Config('config')

kernel = get_kernel(config.kernels.kernel, config.kernels.args.values())

train_data, train_labels = get_sets(config.data.path, "tr", only=0)
test_data, test_ids = get_sets(config.data.path, "te", merge=True)

train_data = kernel.embed(train_data)
test_data = kernel.embed(test_data)
# TODO: 3-fold instead of merging everything

train_data, train_labels, val_data, val_labels = split_train_val(train_data, train_labels, ratio=0.8)

clf = get_classifier(config.classifiers.classifier, kernel, config.classifiers.args.values())
print(clf.fit(train_data, train_labels))

results = clf.evaluate(val_data, val_labels)
print(results)

predictions = clf.predict(test_data)
print(predictions)

save_submission(config, predictions, test_ids, results['Accuracy'])
