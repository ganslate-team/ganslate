from __future__ import print_function
import torch
import numpy as np
import os
from collections import OrderedDict

def remove_module_from_ordered_dict(ordered_dict):
    new_ordered_dict = OrderedDict()
    for k, v in ordered_dict.items():
        if k.startswith('module.'):
            name = k[7:]
            new_ordered_dict[name] = v
        else:
            new_ordered_dict[k] = v
    return new_ordered_dict

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
