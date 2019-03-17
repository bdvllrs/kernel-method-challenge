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
    if type(data) == list:
        # returns couples of the form ("[k]", val)
        # useful for the config class where you can access a list element with the path
        # "data1.data2.[2].item"
        # for config.data1.data2[2].item
        elems = [(f"[{k}]", x) for k, x in list(enumerate(data))]
    else:
        elems = data.items()
    for key, val in elems:
        if key == "gridsearch":
            keys.append((current_path[:-1], val))
        else:
            if type(val) in [dict, list]:
                keys.extend(walk_dict(val, f"{current_path}{key}."))
    return keys


class GridSearch:
    def __init__(self, config):
        self.params = [{}, {}, {}]
        self.all_states = [[], [], []]
        self.config = config
        self.load_from_config()
        self.set_all_states()

    def add(self, set, key, value):
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
        self.params[set][key] = value

    def load_from_config(self):
        """
        Add gridsearch parameters from config
        """
        for i in range(3):
            cfg = self.config[f"set{i}"].values_()
            for key, val in walk_dict(cfg):
                self.add(i, key, val)

    def set_all_states(self):
        for i in range(3):
            keys = list(self.params[i].keys())
            n = len(keys)
            lists = self.params[i].values()
            for element in list_product(*lists):
                self.all_states[i].append({keys[k]: element[k] for k in range(n)})

    def states(self, i):
        """
        Generator of the gridsearch and returns message of what has been set.
        """
        if not self.all_states[i]:
            yield []
        else:
            print(f"Gridsearch: {len(self.all_states[i])} iterations to perform.")
            for k, state in enumerate(self.all_states[i]):
                print(f"\n@@@@@@@ Gridsearch step {k + 1} over {len(self.all_states[i])}: \n")
                print("@@@@@@@ Start model with values: \n")
                self.print_state(state, i, "@@@@@@@")
                yield state

    def print_state(self, state, i=None, prefix=""):
        text = ""
        for key, val in state.items():
            if i is not None:
                self.config[f"set{i}"].set_(key, val)
            text += f"{prefix} - {key}: {val}\n"
        print(text)
