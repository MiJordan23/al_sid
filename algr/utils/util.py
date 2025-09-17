import re
from dataclasses import fields
from typing import Any, Union


def to_bool(data: Any):
    return str(data).lower() in ["true", "on"]


def remove_comments_from_json_string(json_string):
    json_string = re.sub(r'//.*', '', json_string)
    json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
    return json_string

def convert_args_value_type(args, args_class):
    int_fields = []
    float_fields = []
    bool_fields = []
    for field in fields(args_class):
        if field.type == int or field.type == Union[int, None]:
            int_fields.append(field.name)
        elif field.type == float or field.type == Union[float, None]:
            float_fields.append(field.name)
        elif field.type == bool or field.type == Union[bool, None]:
            bool_fields.append(field.name)

    args_converted = {}
    for key, value in args.items():
        if value is not None:
            if key in int_fields and not isinstance(value, int):
                try:
                    value = int(value)
                except Exception as e:
                    raise ValueError(f"convert value of field {key} to int failed, value: {value}, error: {repr(e)}")
            elif key in float_fields and not isinstance(value, float):
                try:
                    value = float(value)
                except Exception as e:
                    raise ValueError(f"convert value of field {key} to float failed, value: {value}, error: {repr(e)}")
            elif key in bool_fields and not isinstance(value, bool):
                try:
                    value = to_bool(value)
                except Exception as e:
                    raise ValueError(f"convert value of field {key} to bool failed, value: {value}, error: {repr(e)}")
        args_converted[key] = value
    return args_converted