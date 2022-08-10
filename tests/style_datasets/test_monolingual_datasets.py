# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from itertools import product

from .utils import *

# Parameters for tests
block_sizes = {
    'eos': [0,6],
    'none': [6,20,1024],
    'complete': [6,20,1024],
    'complete_doc': [10]
}

def get_params(break_mode):
    return [d[break_mode] for d in [
        block_sizes
    ]]

# Tests block sizes lower than some document sizes, higher than document sizes.
# Tests all break modes.
# Tests collater calls on dataset object, and collater calls on list of samples.
# Tests shuffled dataset and non-shuffled.


class TestMonolingualDatasets(unittest.TestCase):

    def sub_test(self, break_mode, dataloader, method, block_size, implicit=True, shuffle=False):
        with self.subTest(msg="break_mode={}, data={}, block_size={}, explicit_samples={}, shuffle={}".format(break_mode, dataloader.__name__, block_size, implicit, shuffle)):
            dataset = build_monolingual_style(*dataloader(), block_size=block_size, break_mode=break_mode, shuffle=shuffle)
            if implicit:
                method(self, dataset.dataset, dataset, indices=None)
            else:
                method(self, dataset.dataset, dataset, indices=np.arange(len(dataset)))

    def test_getitem(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size, implicit_samples) in product([sample1, sample2], *get_params(sbm), [False, True]):
                self.sub_test(sbm, dataloader, check_getitem, block_size, implicit=implicit_samples)

    def test_collater(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size, implicit) in product([sample1, sample2], *get_params(sbm), [False,True]):
                self.sub_test(sbm, dataloader, check_collater, block_size, implicit=implicit)

    def test_collater_with_shuffle(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size) in product([sample1], get_params(sbm)[0]):
                self.sub_test(sbm, dataloader, check_collater, block_size, shuffle=True)

    def test_collater_with_list(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size) in product([sample1], get_params(sbm)[0]):
                self.sub_test(sbm, dataloader, check_collater, block_size, implicit=False)

if __name__ == "__main__":
    unittest.main()
