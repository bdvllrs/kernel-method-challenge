__author__ = "Benjamin Devillers (bdvllrs)"
__version__ = "1.0.4"

import os
import yaml
from re import match


def update_config(conf, new_conf):
    for item in new_conf.keys():
        if type(new_conf[item]) == dict and item in conf.keys():
            conf[item] = update_config(conf[item], new_conf[item])
        else:
            conf[item] = new_conf[item]
    return conf


def map_conf_to_list_elements(list_of_vals):
    """
    When a list is encountered, transform the elements into confs
    Args:
        list_of_vals:

    Returns:

    """
    list_of_conf = []
    for val in list_of_vals:
        if type(val) == dict:
            list_of_conf.append(Config(config=val))
        else:
            list_of_conf.append(val)
    return list_of_conf


class Config:
    def __init__(self, path=None, config=None):
        self.__is_none = False
        data = config if config is not None else {}
        self.__children = {}
        if path is not None:
            self.__path = os.path.abspath(os.path.join(os.curdir, path))
            with open(os.path.join(self.__path, "default.yaml"), "rb") as default_config:
                data.update(yaml.load(default_config))
            for config in sorted(os.listdir(self.__path)):
                if config != "default.yaml" and config[-4:] in ["yaml", "yml"]:
                    with open(os.path.join(self.__path, config), "rb") as config_file:
                        data = update_config(data, yaml.load(config_file))
        self.__type = type(data)
        self.build_conf_obj(data)

    def build_conf_obj(self, data):
        if self.__type == dict:
            elems = data.items()
        elif self.__type == list:
            elems = [(f"[{k}]", x) for k, x in list(enumerate(data))]
        else:  # scalar
            self.__children = data
            return
        for key, val in elems:
            sub_conf = Config(config=val)
            self.__children[key] = sub_conf

    def set_(self, path, value):
        def set_value(_path, _val, child):
            keys = _path.split(".")
            if len(keys) == 1:
                if type(_val) in [dict, list]:
                    new_conf = Config(config=_val)
                    child[keys[0]].__children = new_conf.__children
                    child[keys[0]].__type = new_conf.__type
                else:
                    child[keys[0]].__children = _val
                    child[keys[0]].__type = type(_val)
                return child
            else:
                key = keys.pop(0)
                child[key].__children = set_value(".".join(keys), _val, child[key].__children)
                return child
        self.__children = set_value(path, value, self.__children)

    def values_(self):
        """
        Transform into python data
        """
        if self.__type not in [dict, list]:
            return self.__children
        if self.__type == dict:
            data = {}
            for key, value in self.__children.items():
                data[key] = value.values_()
            return data
        # if list
        data = []
        for key in sorted(self.__children.keys()):
            data.append(self.__children[key].values_())
        return data

    def save_(self, file):
        file = os.path.abspath(os.path.join(os.curdir, file))
        with open(file, 'w') as f:
            yaml.dump(self.values_(), f)

    def get_(self, path, default=None):
        keys = path.split(".")
        if len(keys) == 1:
            if keys[0] in self.__children:
                return self.__children[keys[0]].values_()
            return default
        key = keys.pop(0)
        return self[key].get_(".".join(keys))

    def rename_(self, cur_name, new_name):
        self.__children[new_name] = self.__children.pop(cur_name)

    def __getattr__(self, item):
        # print(item)
        # print(self.__children)
        assert type(self.__children) == dict, "Use the values_() method to get the item."
        if item not in self.__children.keys():
            raise ValueError(f"The item `{item}` does not exist. Use the get_(item, default) method to use a default value.")
        if type(self.__children[item].__children) == dict:
            return self.__children[item]
        return self.__children[item].values_()

    def __getitem__(self, item):
        if type(item) == int:
            item = "[" + str(item) + "]"
        return self.__getattr__(item)

    def __str__(self):
        if self.__type not in [dict, list]:
            return str(self.__children)
        elems = []
        for key, val in self.__children.items():
            if self.__type == list:
                elems.append(val.__str__())
            else:
                elems.append(key + ": " + val.__str__())
        if self.__type == dict:
            return "{" + ", ".join(elems) + "}"
        else:
            return "[" + ", ".join(elems) + "]"
