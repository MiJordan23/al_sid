import copy
import json
from typing import Union

class EasyDict(dict):
    def __init__(self, dict_or_path: Union[dict, str]):
        if isinstance(dict_or_path, dict):
            super().__init__(copy.deepcopy(dict_or_path))
        elif isinstance(dict_or_path, str):
            with open(dict_or_path, 'r') as f:
                config = EasyDict(json.load(f))
            super().__init__(config)
        else:
            raise TypeError(f'dict_or_path must be a dict or a path to a json file, current is {type(dict_or_path)}.')

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            try:
                return object.__getattribute__(self, item)
            except AttributeError as e:
                if item.startswith('__'):
                    raise e
                else:
                    return None

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        return dict(self)
