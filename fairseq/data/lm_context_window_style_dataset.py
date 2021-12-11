# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import copy

from fairseq.data.indexed_dataset import MMapIndexedDataset
from fairseq.data.token_block_dataset import TokenBlockDataset
from . import LMContextWindowDataset

import sys
sys.path.append("../..")
from style_knnlm.utils import (
    pick_layer
    , sample_to_token_slices
    , get_doc_sizes, get_doc_gaps
    , _get_block_to_dataset_index_fast
    , tokens_per_document
)
sys.path.pop()
# TODO: find better solution for this


def context_expansion(index, capacity, sizes):
    """Expand sample with leftward context.
        Takes up to `capacity` tokens if available.
    """

    i = index
    consumed = 0
    offset = 0
    while i > 0:
        remaining = capacity - consumed
        if remaining > 0:
            i -= 1
            offset = sizes[i] - remaining
            if offset > 0:
                consumed += remaining
                break
            else:
                consumed += sizes[i]
        else:
            break

    return [i, max(0,offset), index]

def resample_style(dataset, ids, context_ranges):
    docsizes = get_doc_sizes(dataset)
    docskips = get_doc_gaps(dataset)
    docslices_ext = _get_block_to_dataset_index_fast(
        docsizes + docskips, sample_to_token_slices(dataset, context_ranges)
    )
    tpds = [tokens_per_document(i, dataset, docslices_ext, docsizes) for i in ids]
    avg_style = np.array([
        np.average(dataset.style_attributes[indices], weights=weights) for indices, weights in tpds
    ])

    return avg_style

class LMContextWindowStyleDataset(LMContextWindowDataset):
    def __init__(self, dataset, style_attributes):
        assert isinstance(self, LMContextWindowDataset) # check dataset instead of self ?
        
        self.dataset = dataset
        self.style_attributes = np.array(style_attributes)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def collater(self, samples):
        return Collaters.v4(self, samples)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)


class Collaters:
    # NOTE: original impl., with comments.
    def v1(dataset, samples, precomputed=False):
        if precomputed:
            sample = samples
        else:
            sample = dataset.collater(samples) # token block samples (MonolingualDataset inherits collater)
        
        pad = dataset.pad_idx
        max_sample_len = dataset.tokens_per_sample + dataset.context_window

        bsz, tsz = sample['net_input']['src_tokens'].shape # take previous batch shape.
        start_idxs = [0] * bsz # set start index to 0 for each sample.
        
        # copy previous sample arrays.
        toks = sample['net_input']['src_tokens'] # copy previous sample input tokens.
        lengths = sample['net_input']['src_lengths'] # copy previous sample input lengths.
        tgt = sample['target'] # copy targets.

        # initialize new sample arrays.
        new_toks = np.empty([bsz, tsz + dataset.context_window], dtype=np.int64) # input tokens.
        new_tgt = np.full([bsz, tsz + dataset.context_window], pad, dtype=np.int64) # target tokens.
        sample_lens = toks.ne(pad).long().sum(dim=1).cpu() # input lengths, excluding pad tokens.

        for i in range(bsz):
            sample_len = sample_lens[i] # given sample length

            # if previous tokens and sample too long, drop extra previous tokens
            extra = len(dataset.prev_tokens) + sample_len - max_sample_len 
            if extra > 0:
                dataset.prev_tokens = dataset.prev_tokens[extra:]

            # Assign new input tokens and target tokens.
            # Input tokens are right-padded if context+sample are not long enough.
            # Target is left-padded.
            pads = np.full(dataset.context_window - len(dataset.prev_tokens), pad)
            new_toks[i] = np.concatenate([dataset.prev_tokens, toks[i].numpy(), pads])
            new_tgt[i, len(dataset.prev_tokens):len(dataset.prev_tokens) + len(tgt[i])] = tgt[i] 

            start_idxs[i] = len(dataset.prev_tokens) # index where added context tokens end.
            lengths[i] += len(dataset.prev_tokens) # increase length by number of added context tokens.

            dataset.prev_tokens = new_toks[i][new_toks[i] != pad][-dataset.context_window:] # use non-pad, non-context tokens from current input tokens as previous for next sample.
        sample['net_input']['src_tokens'] = torch.from_numpy(new_toks)
        sample['target'] = torch.from_numpy(new_tgt)
        sample['start_indices'] = start_idxs

        return sample

    # NOTE: WIP
    def v4(dataset, batch, debug=False):
        assert isinstance(dataset, LMContextWindowStyleDataset)

        pad = dataset.dataset.pad_idx
        max_sample_len = dataset.dataset.tokens_per_sample + dataset.dataset.context_window

        if debug:
            samples = copy.deepcopy(batch)
            # NOTE: necessary to avoid overwriting batches 
            # when (re)using precomputed batches during development and testing.
        else:
            samples = batch

        if not isinstance(samples, dict):
            print("Collating.")
            samples = dataset.dataset.dataset.collater(samples) # collate in superclass.
            # NOTE: theoretically collating could be skipped since id, source and target 
            # can be accessed by ids in batch.

        bsz, tsz = samples['net_input']['src_tokens'].shape # take batch shape.
        start_idxs = np.zeros([bsz], dtype=int) # offsets of original samples inside expanded samples
        
        # copy sample arrays in batch.
        ids = samples["id"].numpy()
        toks = samples['net_input']['src_tokens']
        tgts = samples['target']

        # initialize sample arrays for new batch.
        new_toks = np.full([bsz, tsz + dataset.dataset.context_window], pad, dtype=np.int64) # input tokens.
        new_tgts = np.full([bsz, tsz + dataset.dataset.context_window], pad, dtype=np.int64) # target tokens.
        sample_lens = dataset.dataset.dataset.sizes[ids] # TODO: A/B test.
        #sample_lens = toks.ne(pad).long().sum(dim=1).cpu() # input lengths, excluding pad tokens.
        tgts_lens = tgts.ne(pad).long().sum(dim=1).cpu() # input lengths, excluding pad tokens.

        # max number of tokens each sample can be expanded by.
        capacity = np.minimum(max_sample_len - sample_lens, dataset.dataset.context_window)
        context_ranges = np.array([context_expansion(i, capacity[i], dataset.dataset.dataset.sizes) for i in ids])

        for i in range(bsz):
            a,offset,b = context_ranges[i]

            if b > a:
                context_tokens = np.concatenate([
                    dataset.dataset[j]["source"] 
                    for j in range(a,b)
                ])[offset:] # concatenate all context samples and drop leftover tokens
            else:
                context_tokens = np.array([], dtype=np.int64)

            context_len = len(context_tokens)
            start_idxs[i] = context_len

            if context_len > 0:
                new_toks[i, 0:context_len] = context_tokens # insert context if existing
            new_toks[i, context_len:context_len+sample_lens[i]] = toks[i][toks[i] != pad].numpy() 
            new_tgts[i, context_len:context_len+tgts_lens[i]] = tgts[i][tgts[i] != pad].numpy()

        # TODO: rework this part. 
        # Either compute everything once, outside of this function, or fully support element-wise operations.
        sample_lens = dataset.dataset.sizes
        capacity = np.minimum(max_sample_len - sample_lens, dataset.dataset.context_window)
        full_context_ranges = np.array([context_expansion(i, capacity[i], dataset.dataset.sizes) for i in range(len(dataset))])
        avg_style = resample_style(dataset, ids, full_context_ranges) # resample style attributes

        samples['net_input']['src_tokens'] = torch.from_numpy(new_toks)
        samples['target'] = torch.from_numpy(new_tgts)
        samples['start_indices'] = start_idxs
        samples['net_input']['src_lengths'] += start_idxs
        samples['style'] = torch.from_numpy(avg_style) # see if torch is even necessary.

        return samples