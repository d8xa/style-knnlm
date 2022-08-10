# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import logging

from fairseq.data.monolingual_dataset import MonolingualDataset
from . import FairseqDataset
import style_knnlm.utils.functions

logger = logging.getLogger(__name__)


class LMContextWindowDataset(FairseqDataset):
    """Wraps a MonolingualDataset and provides more context for evaluation."""

    def __init__(self, dataset, tokens_per_sample, context_window, pad_idx, collater_impl="new"):
        assert isinstance(dataset, MonolingualDataset), "got {}".format(type(dataset).__name__)
        assert context_window > 0
        self.dataset = dataset
        self.tokens_per_sample = tokens_per_sample
        self.context_window = context_window
        self.pad_idx = pad_idx
        self.collater_impl = Collaters.modified # prepare for switching via CLI option
        if collater_impl == "old":
            self.collater_impl = Collaters.original
        if self.collater_impl is Collaters.original:
            self.prev_tokens = np.empty([0])
        else:
            style_knnlm.utils.functions.initialize_index_arrays(self)


    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            dataset: the LMContextWindowDataset object
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `nsentences`: total number of sentences in the batch
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                    - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                        the source sentence of shape `(bsz, src_len)`. Padding will
                        appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                    target sentence of shape `(bsz, tgt_len)`. Padding will appear
                    on the right.
                - `start_indices` (numpy.ndarray): a 1D array of indices denoting 
                    up until which index net input tokens are context tokens.
        """
        return self.collater_impl(self, samples)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    @property
    def sizes(self):
        if not hasattr(self, "_sizes"):
            self._sizes = np.array([self.num_tokens(i) for i in range(len(self))]) #+ self.context_window
        return self._sizes

    def ordered_indices(self):
        # NOTE we don't shuffle the data to retain access to the previous dataset elements
        return np.arange(len(self.dataset))

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)


class Collaters:
    from typing import List, Dict

    def modified(dataset: LMContextWindowDataset, samples: List[Dict]):
        """And index-based version of the original collater function."""

        samples = dataset.dataset.collater(samples)

        # shapes and indices
        B, T = samples['net_input']['src_tokens'].shape # batch and tokens
        C = dataset.context_window # context window
        start_idxs = np.zeros([B], dtype=int) # offsets of original samples inside expanded samples
        pad = dataset.pad_idx
        
        # copy sample arrays in batch.
        ids = samples["id"].numpy()
        toks = samples['net_input']['src_tokens']
        tgts = samples['target']

        # initialize sample arrays for new batch.
        new_toks = np.full([B, T+C], pad, dtype=style_knnlm.utils.functions.as_numpy_dtype(toks.dtype))
        new_tgts = np.full([B, T+C], pad, dtype=style_knnlm.utils.functions.as_numpy_dtype(tgts.dtype))
        sample_lens = dataset.sample_lens[ids] # input lengths, excluding pad tokens.
        tgts_lens = tgts.ne(pad).long().sum(dim=1).cpu() # target lengths, excluding pad tokens.

        for i in range(B):
            a,offset,b = dataset.slice_indices_sample[ids[i]]

            if b > a:
                context_tokens = np.concatenate([
                    dataset.dataset[j]["source"].numpy()
                    for j in range(a,b)
                ])[offset:]
            else:
                context_tokens = np.array([], dtype=new_toks.dtype)

            context_len = len(context_tokens)
            if context_len > 0:
                new_toks[i, 0:context_len] = context_tokens
                start_idxs[i] = context_len
            elif ids[i] == 0:
                start_idxs[i] = C
            new_toks[i, context_len:context_len+sample_lens[i]] = toks[i][toks[i] != pad].numpy() 
            new_tgts[i, context_len:context_len+tgts_lens[i]] = tgts[i][tgts[i] != pad].numpy()
        
        samples['net_input']['src_tokens'] = torch.from_numpy(new_toks)
        samples['target'] = torch.from_numpy(new_tgts)
        samples['start_indices'] = start_idxs
        samples['net_input']['src_lengths'] += start_idxs

        return samples

    def original(dataset, samples):
        sample = dataset.dataset.collater(samples)

        pad = dataset.pad_idx
        max_sample_len = dataset.tokens_per_sample + dataset.context_window

        bsz, tsz = sample['net_input']['src_tokens'].shape
        start_idxs = [0] * bsz
        toks = sample['net_input']['src_tokens']
        lengths = sample['net_input']['src_lengths']
        tgt = sample['target']
        new_toks = np.empty([bsz, tsz + dataset.context_window], dtype=np.int64)
        new_tgt = np.full([bsz, tsz + dataset.context_window], pad, dtype=np.int64)
        sample_lens = toks.ne(pad).long().sum(dim=1).cpu()

        for i in range(bsz):
            sample_len = sample_lens[i]
            extra = len(dataset.prev_tokens) + sample_len - max_sample_len
            if extra > 0:
                dataset.prev_tokens = dataset.prev_tokens[extra:]
            pads = np.full(dataset.context_window - len(dataset.prev_tokens), pad)
            new_toks[i] = np.concatenate([dataset.prev_tokens, toks[i].numpy(), pads])
            new_tgt[i, len(dataset.prev_tokens):len(dataset.prev_tokens) + len(tgt[i])] = tgt[i]
            start_idxs[i] = len(dataset.prev_tokens)
            lengths[i] += len(dataset.prev_tokens)
            dataset.prev_tokens = new_toks[i][new_toks[i] != pad][-dataset.context_window:]

        sample['net_input']['src_tokens'] = torch.from_numpy(new_toks)
        sample['target'] = torch.from_numpy(new_tgt)
        sample['start_indices'] = start_idxs

        return sample