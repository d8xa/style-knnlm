# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import logging

from . import FairseqDataset
from fairseq.data.monolingual_style_dataset import MonolingualStyleDataset
import style_knnlm.utils

logger = logging.getLogger(__name__)

class LMContextWindowStyleDataset(FairseqDataset):
    def __init__(self, dataset: MonolingualStyleDataset, tokens_per_sample, context_window, pad_idx):
        self.dataset = dataset
        self.context_window = context_window
        self.tokens_per_sample = tokens_per_sample
        self.pad_idx = pad_idx
        style_knnlm.utils.initialize_index_arrays(self)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return Collaters.collate(self, samples)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    @property
    def sizes(self):
        return self.dataset.sizes


class Collaters:
    from typing import List, Dict

    def collate(dataset: MonolingualStyleDataset, samples: List[Dict]):
        samples = dataset.dataset.collater(samples)

        # shapes and indices
        B, T = samples['net_input']['src_tokens'].shape # batch and tokens
        C = dataset.context_window # context window
        S = samples['style'].shape[-1] # style dimension
        start_idxs = np.zeros([B], dtype=int) # offsets of original samples inside expanded samples
        pad = dataset.pad_idx
        
        # copy sample arrays in batch.
        ids = samples["id"].numpy()
        toks = samples['net_input']['src_tokens']
        tgts = samples['target']
        style = samples['style']

        # initialize sample arrays for new batch.
        new_toks = np.full([B, T+C], pad, dtype=style_knnlm.utils.as_numpy_dtype(toks.dtype))
        new_tgts = np.full([B, T+C], pad, dtype=style_knnlm.utils.as_numpy_dtype(tgts.dtype))
        new_stls = np.full([B, T+C, S], style.mean(dim=1, keepdim=True), dtype=style_knnlm.utils.as_numpy_dtype(style.dtype))
        # NOTE: for a lack of a better solution the style array is filled with the mean style value per sample. This might change in the future if an appropriate representation of neutral style is introduced.
        sample_lens = dataset.sample_lens[ids] # input lengths, excluding pad tokens.
        tgts_lens = tgts.ne(pad).long().sum(dim=1).cpu() # target lengths, excluding pad tokens.

        for i in range(B):
            a,offset,b = dataset.slice_indices_sample[ids[i]]

            if b > a:
                context_tokens = np.concatenate([
                    dataset.dataset[j]["source"].numpy()
                    for j in range(a,b)
                ])[offset:]
                context_styles = np.concatenate([
                    dataset.dataset[j]["style"].numpy()
                    for j in range(a,b)
                ])[offset:]
            else:
                context_tokens = np.array([], dtype=new_toks.dtype)
                context_styles = np.array([], dtype=new_stls.dtype)

            context_len = len(context_tokens)
            if context_len > 0:
                new_toks[i, 0:context_len] = context_tokens
                new_stls[i, 0:context_len] = context_styles
                start_idxs[i] = context_len
            elif ids[i] == 0: 
                # special case for first sample where context is empty.
                start_idxs[i] = C
            new_toks[i, context_len:context_len+sample_lens[i]] = toks[i][toks[i] != pad].numpy() 
            new_tgts[i, context_len:context_len+tgts_lens[i]] = tgts[i][tgts[i] != pad].numpy()
            new_stls[i, context_len:context_len+sample_lens[i]] = style[i][toks[i] != pad].numpy()
        
        samples['net_input']['src_tokens'] = torch.from_numpy(new_toks)
        samples['target'] = torch.from_numpy(new_tgts)
        samples['start_indices'] = start_idxs
        samples['net_input']['src_lengths'] += start_idxs
        samples['style'] = torch.from_numpy(new_stls)

        return samples