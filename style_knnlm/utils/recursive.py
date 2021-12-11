import torch
import numpy as np


def deep_shape(d):
    shapes = {}
    for k in d.keys():
        if isinstance(d[k], dict):
            shapes[k] = deep_shape(d[k])
        elif isinstance(d[k], (torch.Tensor, np.ndarray)):
            shapes[k] = d[k].shape
        elif isinstance(d[k], (tuple, list)):
            shapes[k] = len(d[k])
    return shapes

def deep_equals(d1, d2):
    # NOTE: assume equal keys and equal value types
    mismatches = {}
    for k in d1.keys():
        if isinstance(d1[k], dict):
            mismatches.update(deep_equals(d1[k], d2[k]))
        elif isinstance(d1[k], (list, torch.Tensor, np.ndarray)):
            if not (d1[k] == d2[k]).all():
                mismatches[k] = d1[k] == d2[k]
        elif not d1[k] == d2[k]:
                mismatches[k] = True
    return mismatches