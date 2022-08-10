# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from itertools import product
from collections import defaultdict

from fairseq.data.lm_context_window_dataset import Collaters

from .utils import *


# Parameters for tests
block_sizes = {
    'eos': [0,6],
    'none': [6,20],
    'complete': [6,20],
    'complete_doc': [10]
}
context_sizes = defaultdict(lambda: [3], {})
tokens_per_sample = defaultdict(lambda: [6], {})

def get_params(break_mode):
    return [d[break_mode] for d in [
        block_sizes, 
        context_sizes, 
        tokens_per_sample
    ]]


class TestContextDatasets(unittest.TestCase):

    def sub_test(self, break_mode, dataloader, method, block_size, context_window, tokens_per_sample, shuffle=False, implicit=True):
        with self.subTest(msg="break_mode={}, data={}, block_size={}, cw={}, tps={}".format(break_mode, dataloader.__name__, block_size, context_window, tokens_per_sample)):
            dataset1 = build_lmcw(dataloader()[0], block_size=block_size, break_mode=break_mode, context_window=context_window, tokens_per_sample=tokens_per_sample, shuffle=shuffle)
            dataset2 = build_lmcws(*dataloader(), block_size=block_size, break_mode=break_mode, context_window=context_window, tokens_per_sample=tokens_per_sample, shuffle=shuffle)
            if implicit:
                method(self, dataset1, dataset2, indices=None)
            else:
                method(self, dataset1, dataset2, indices=np.arange(len(dataset1)))

    def test_getitem(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size, cw, tps) in product([sample1, sample2], *get_params(sbm)):
                self.sub_test(sbm, dataloader, check_getitem, block_size, cw, tps)

    def test_collater(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size, cw, tps) in product([sample1, sample2], *get_params(sbm)):
                self.sub_test(sbm, dataloader, check_collater, block_size, cw, tps)

    def test_collater_with_shuffle(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size, cw, tps) in product([sample1], *[x[0:1] for x in get_params(sbm)]):
                self.sub_test(sbm, dataloader, check_collater, block_size, cw, tps, shuffle=True)

    def test_collater_with_list(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size, cw, tps) in product([sample1], *[x[0:1] for x in get_params(sbm)]):
                self.sub_test(sbm, dataloader, check_collater, block_size, cw, tps, implicit=False)

    def test_new_vs_old(self):
        for sbm in ["eos", "none", "complete", "complete_doc"]:
            for (dataloader, block_size, cw, tps) in product([sample1], *[x[0:1] for x in get_params(sbm)]):
                new = build_lmcw(dataloader()[0], block_size=block_size, break_mode=sbm, context_window=cw, tokens_per_sample=tps, shuffle=True)
                old = build_lmcw(dataloader()[0], block_size=block_size, break_mode=sbm, context_window=cw, tokens_per_sample=tps, shuffle=True, collater_impl="old")
                assert old.collater_impl == Collaters.original
                check_getitem(self, old, new, style=False)
                check_collater(self, old, new, style=False)

if __name__ == "__main__":
    unittest.main()
