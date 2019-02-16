import pandas as pd
from utils import Config, get_classifier, get_sets, get_kernel, split_train_val

config = Config('config')

kernel = get_kernel(config.kernels.kernel, config.kernels.args.as_dict())

train_sets = get_sets(config.data.path, "tr")
# Combine all sets as one dataset
# TODO: 3-fold
train_set = pd.concat(train_sets)
# Add a "vec" column as kernel embedding
train_set["vec"] = train_set.apply(lambda x: kernel.embed(x['seq']), axis=1)

train_set, val_set = split_train_val(train_set, ratio=0.8)
test_set = get_sets(config.data.path, "te")[0]

clf = get_classifier(config.classifiers.classifier, train_set, kernel, config.classifiers.args.as_dict())

# clf.fit()

# clf.evaluate(val_set)

# clf.predict(test_set)
