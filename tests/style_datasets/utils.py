import torch as tc
import numpy as np

import tests.utils as test_utils

from fairseq.data import (
    TokenBlockDataset, 
    MonolingualDataset, 
    MonolingualStyleDataset, 
    LMContextWindowDataset, 
    LMContextWindowStyleDataset
)

# collater
def net_input(sample): return sample["net_input"]
def src_style(sample): return sample["style"]

# getitem
def src(sample): 
    if "net_input" in sample:
        return sample["net_input"]["src_tokens"]
    return sample["source"]
def tgt(sample): return sample["target"]

def sample1():
    """A small sample of truncated sentences from the Wikitext-103 dataset. """
    data = [
        tc.LongTensor([2]),
        tc.LongTensor([12, 52547, 11408, 1144, 12, 2]),
        tc.LongTensor([2]),
        tc.LongTensor([159046, 129, 52547, 94, 45, 3, 11408, 25, 466, 45]),
        tc.LongTensor([16, 84, 142, 425, 10, 289, 5, 2384, 73]),
        tc.LongTensor([66, 708, 22, 754, 1205, 10, 663]),
        tc.LongTensor([2]),
        tc.LongTensor([12, 12, 7699, 12, 12, 2]),
        tc.LongTensor([2]),
        tc.LongTensor([149, 22, 441, 3, 11408, 260, 5, 52547, 11408, 1144, 26, 11, 6753, 297]),
        tc.LongTensor([2])
    ]
    style = np.array([
        0.06973077, 0.08137056, -0.02129515, 0.0732347, 0.11086446, 0.08522498
    ]).reshape(-1,1)
    return data, style

def sample2():
    """A small sample of truncated sentences from the Stanford Politeness Corpus."""
    data = [
        tc.LongTensor([252, 879, 1093, 740, 845, 873, 9, 137, 730, 20396, 55, 5, 173, 15, 177, 14, 2944, 2]),
        tc.LongTensor([133, 602, 11, 5, 345, 54, 400, 13, 926, 19, 109, 659, 8, 222, 617, 2]),
        tc.LongTensor([226, 15823, 4, 1304, 9, 16300, 60, 10, 10, 2113, 84, 89, 22015, 2]),
        tc.LongTensor([3919, 38, 6619, 45, 616, 6751, 11, 19280, 88, 3064, 53, 13340, 183, 17, 114, 6, 212, 8373, 145, 2226, 48, 19281, 71, 35, 139, 4, 397, 207, 97, 24430, 86, 5, 298, 11, 26779, 4255, 20, 5, 2215, 45, 47, 21290, 19654, 2]),
        tc.LongTensor([8, 368, 27, 13459, 19, 21, 1171, 354, 7, 1760, 1012, 22, 5, 310, 9, 83, 12897, 3005, 4477, 6, 20, 167, 2]),
        tc.LongTensor([64, 5, 338, 690, 4, 144, 1167, 3438, 10, 11, 7, 4947, 2]),
        tc.LongTensor([9353, 328, 6, 1284, 20, 8, 113, 54, 1409, 14, 4, 22608, 40, 4, 22146, 17762, 39, 65, 174, 12, 491, 6, 145, 7, 1329, 25508, 2]),
        tc.LongTensor([1779, 5, 158, 8713, 10, 7, 7982, 20, 5, 61, 89, 198, 7, 16113, 138, 97, 17717, 178, 562, 6, 302, 4149, 17801, 2]),
        tc.LongTensor([87, 540, 76, 6, 20, 45, 9, 11, 804, 14, 98, 31, 16, 13775, 486, 73, 7, 331, 26197, 27339, 6725, 39, 5, 147, 93, 2]),
        tc.LongTensor([21, 10, 25, 22349, 21, 10, 5024, 3990, 24, 18, 153, 1108, 10070, 17138, 10544, 968, 2])
    ]
    style = np.array([
        1.00773697, 0.60744342, 0.2938268, 0.69778127, 0.97238147, -0.05576258, 0.06636316, 0.09505936, -0.33382334, -0.48085676
    ]).reshape(-1,1)
    return data, style

def check_getitem(test, dataset1, dataset2, indices=None, style=True):
        if indices is None:
            indices = range(len(dataset1))
        for i in indices:
            sample1 = dataset1[i]
            sample2 = dataset2[i]
            test.assertTrue(tc.equal(src(sample1), src(sample2)))
            test.assertTrue(tc.equal(tgt(sample1), tgt(sample2)))
            test.assertSequenceEqual(src(sample1).shape, src(sample2).shape)
            test.assertSequenceEqual(tgt(sample1).shape, tgt(sample2).shape)
            if style:
                test.assertSequenceEqual(src(sample2).shape, src_style(sample2).shape[:-1])

def check_collater(test, dataset1, dataset2, indices=None, style=True):
    if indices is None:
        sample1 = dataset1.collater(dataset1)
        sample2 = dataset2.collater(dataset2)
    else:
        sample1 = dataset1.collater([dataset1[i] for i in indices])
        sample2 = dataset2.collater([dataset2[i] for i in indices])

    test.assertTrue(tc.equal(src(sample1), src(sample2)))
    test.assertTrue(tc.equal(tgt(sample1), tgt(sample2)))
    test.assertSequenceEqual(src(sample1).shape, src(sample2).shape)
    test.assertSequenceEqual(tgt(sample1).shape, tgt(sample2).shape)
    if style:
        test.assertSequenceEqual(src(sample2).shape, src_style(sample2).shape[:-1])


def build_tokenblock(data, block_size, break_mode, vocab=None):
    sizes = np.array([len(x) for x in data])
    dataset = test_utils.TestDataset(data)
    dataset.sizes = sizes
    if vocab is None:
        vocab = test_utils.dummy_dictionary(max([max(x) for x in data]))
    dataset = TokenBlockDataset(
        dataset=dataset, 
        sizes=dataset.sizes, 
        block_size=block_size, 
        pad=vocab.pad_index, 
        eos=vocab.eos_index, 
        break_mode=break_mode, 
        include_targets=True
    )
    return dataset

def build_monolingual(data, block_size, break_mode, shuffle=False):
    vocab = test_utils.dummy_dictionary(max([max(x) for x in data]))
    dataset = build_tokenblock(data, block_size, break_mode, vocab=vocab)
    dataset = MonolingualDataset(
        dataset=dataset, 
        sizes=dataset.sizes,
        src_vocab=vocab, 
        tgt_vocab=vocab, 
        add_eos_for_other_targets=(break_mode != "none"), 
        shuffle=shuffle, 
        targets=['future']
    )
    return dataset

def build_monolingual_style(data, style, block_size, break_mode, shuffle=False):
    dataset = build_monolingual(data, block_size, break_mode, shuffle=shuffle)
    return MonolingualStyleDataset(dataset, style)

def build_lmcw(data, block_size, break_mode, context_window, tokens_per_sample, shuffle=False, collater_impl="new"):
    dataset = build_monolingual(data, block_size, break_mode, shuffle=shuffle)
    dataset = LMContextWindowDataset(
        dataset=dataset,
        tokens_per_sample=tokens_per_sample,
        context_window=context_window,
        pad_idx=dataset.vocab.pad(),
        collater_impl=collater_impl
    )
    return dataset

def build_lmcws(data, style, block_size, break_mode, context_window, tokens_per_sample, shuffle=False):
    dataset = build_monolingual_style(data, style, block_size, break_mode, shuffle=shuffle)
    dataset = LMContextWindowStyleDataset(
        dataset=dataset, 
        tokens_per_sample=tokens_per_sample, 
        context_window=context_window, 
        pad_idx=dataset.dataset.vocab.pad()
    )
    return dataset