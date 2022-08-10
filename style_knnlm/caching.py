from style_knnlm.evaluation.flags import VariablesFlags

import numpy as np
import torch as tc
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

class StoredVariable(list):
    """Represents a temporarily stored variable."""

    def __init__(self, name=None):
        super(StoredVariable, self).__init__()
        self.name = name

    def peek(self):
        if len(self) > 0:
            return self[-1]
        return None

    def push(self, value):
        self.append(value)

    def save(self, path):
        if len(self) == 0:
            return
        if isinstance(self[-1], np.ndarray):
            np.save(
                file=Path(path).joinpath("{}.npy".format(self.name)),
                arr=np.concatenate(self)
            )
        elif tc.is_tensor(self[-1]):
            tc.save(
                tc.concat(self), 
                Path(path).joinpath("{}.pt".format(self.name))
            )
        elif tc.is_tensor(self):
            tc.save(self, Path(path).joinpath("{}.pt".format(self.name)))
        else:
            raise NotImplementedError("No method defined to save cache \"{}\" of type {}.".format(self.name, type(self[-1]).__name__))

class BufferedVariable(list):
    """Represents a buffer of size 1."""

    def __init__(self, name=None):
        super(BufferedVariable, self).__init__()
        self.name = name

    def peek(self):
        if len(self) > 0:
            return self[-1]
        return None

    def push(self, value):
        if len(self) == 0:
            self.append(value)
        else:
            self[-1] = value
    
    def save(self, _):
        pass

class CachingManager():
    """A manager for variables cached during LM/DS evaluation."""

    def __init__(self, flags = VariablesFlags.NONE, persist = VariablesFlags.NONE):
        self.all = VariablesFlags.NONE
        self.persist = VariablesFlags.NONE # which to save to disk later
        self.caches = {}
        self._register_flags(flags, persist)

    def _register_flags(self, flags, persist):
        for flag in VariablesFlags:
            if (flags & flag):
                if (flag & persist):
                    self.persist |= flag
                    self.caches[flag] = StoredVariable(
                        name=flag.name.lower()+"_cache"
                    )
                else:
                    self.caches[flag] = BufferedVariable(
                        name=flag.name.lower()+"_cache"
                    )
                self.all |= flag

    def push(self, flag, value):
        self.caches[flag].push(value)

    def peek(self, flag):
        return self.caches[flag].peek()

    def pop(self, flag):
        return self.caches[flag].pop()

    def write(self, path):
        if len(self.caches) == 0:
            return
        
        if Path(path).is_file():
            raise ValueError('Path is not a directory')
        Path(path).mkdir(exist_ok=True, parents=True)
        
        for cache in self.caches.values():
            cache.save(path)