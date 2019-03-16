import numpy as np
from itertools import product as list_product


def walk_dict(data: dict, current_path=""):
    """
    Returns all full key path from a dict looking for gridsearch
    Args:
        current_path:
        data: dict data to walk

    Returns: generator

    Examples:
        >>> data = {"test": {"gridsearch": [0, 1, 2]}, "one": {"two": {"gridsearch": [4, 5, 6]}}}
        >>> walk_dict(data)
        [("test", [0, 1, 2]), ("one.two", [4, 5, 6])]
    """
    keys = []
    for key, val in data.items():
        if key == "gridsearch":
            keys.append((current_path[:-1], val))
        else:
            if type(val) == dict:
                keys.extend(walk_dict(val, f"{current_path}{key}."))
    return keys


class GridSearch:
    def __init__(self, config):
        self.params = {}
        self.all_states = []
        self.config = config
        self.load_from_config()
        self.set_all_states()

    def add(self, key, value):
        # If it's a dict, we transform it into a list
        if type(value) == dict:
            list_type = value["type"] if "type" in value.keys() else "linear"
            assert "num" in value.keys(), f"len is not defined for gridsearch {key}"
            assert "min" in value.keys(), f"min is not defined for gridsearch {key}"
            assert "max" in value.keys(), f"max is not defined for gridsearch {key}"
            if list_type == "log":
                value = np.logspace(value["min"], value["max"], value["num"])
            else:  # linear
                value = np.linspace(value["min"], value["max"], value["num"])
        self.params[key] = value

    def load_from_config(self):
        """
        Add gridsearch parameters from config
        """
        cfg = self.config.values_()
        for key, val in walk_dict(cfg):
            self.add(key, val)

    def set_all_states(self):
        keys = list(self.params.keys())
        n = len(keys)
        lists = self.params.values()
        for element in list_product(*lists):
            self.all_states.append({keys[k]: element[k] for k in range(n)})

    def states(self):
        """
        Generator of the gridsearch and returns message of what has been set.
        """
        print(f"Gridsearch: {len(self.all_states)} iterations to perform.")
        for k, state in enumerate(self.all_states):
            print(f"\n@@@@@@@ Gridsearch step {k + 1} over {len(self.all_states)}: \n")
            print("@@@@@@@ Start model with values: \n")
            self.print_state(state, "@@@@@@@")
            yield state

    def print_state(self, state, prefix=""):
        text = ""
        for key, val in state.items():
            self.config.set_(key, val)
            text += f"{prefix} - {key}: {val}\n"
        print(text)
