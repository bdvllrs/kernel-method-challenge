from utils import Config, get_classifier, get_sets, get_kernel, split_train_val

config = Config('config')

kernel = get_kernel(config.kernels.kernel, config.kernels.args.as_dict())

train_data, train_labels = get_sets(config.data.path, "tr")
test_data = get_sets(config.data.path, "te", merge=True)

train_data = kernel.embed(train_data[0])
test_data = kernel.embed(test_data)
# TODO: 3-fold instead of merging everything

train_data, train_labels, val_data, val_labels = split_train_val(train_data, train_labels[0], ratio=0.8)

clf = get_classifier(config.classifiers.classifier, kernel, config.classifiers.args.as_dict())

print("Fitting...")
print(clf.fit(train_data, train_labels))

print("Evaluating...")
print(clf.evaluate(val_data, val_labels))

print("Predicting...")
print(clf.predict(test_data))
