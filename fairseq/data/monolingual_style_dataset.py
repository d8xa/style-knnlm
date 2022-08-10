# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from typing import Iterable, List

from . import FairseqDataset
from fairseq.data.monolingual_dataset import MonolingualDataset

import style_knnlm.utils



def collate_style(values: List[torch.Tensor]
    , pad_value=None 
    # TODO: find good solution for pad_value. 
    # Currently using mean style value per sample.
    , left_pad=False
    ):
    """Version of `data_utils.collate_tokens` for style values."""

    size = max(len(v) for v in values)
    res = values[0].new_empty(len(values), size, values[0].shape[-1])

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i,v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        if left_pad:
            res[i][:size-len(v)] = v.mean(axis=0, keepdim=True) if pad_value is None else pad_value
        else:
            res[i][len(v):] = v.mean(axis=0, keepdim=True) if pad_value is None else pad_value

    return res
    

def get_style_indices(sample_ids: Iterable, dataset):
    """Returns the associated style attributes for each token in the samples at indices ``sample_ids``."""
    
    return [
        np.repeat(*style_knnlm.tokens_per_document(sid, dataset, dataset.s2d_slices, dataset.docsizes)) 
        for sid in sample_ids
    ]


class MonolingualStyleDataset(FairseqDataset):

    def __init__(self, dataset: MonolingualDataset, style: Iterable):
        self.__dict__ = {k:v for k,v in dataset.__dict__.items() if k != "dataset"}
        self.dataset = dataset
        self.style = style

        self.gaps_included = dataset.sizes.sum() != style_knnlm.utils.DatasetTools.token_counts(dataset).sum()

        token_counts = style_knnlm.utils.DatasetTools.token_counts(dataset)
        doc_slices = style_knnlm.utils.DatasetTools.doc_slices(dataset, token_counts=token_counts)
        docsizes = style_knnlm.utils.DatasetTools.doc_sizes(dataset, doc_slices=doc_slices)
        self.docntokens = docsizes + style_knnlm.utils.DatasetTools.doc_skips(self.dataset, doc_slices=doc_slices, token_counts=token_counts)
        self.s2d_slices = style_knnlm.utils._get_block_to_dataset_index_fast(self.docntokens, style_knnlm.utils.DatasetTools.block_slices(self.dataset))
        self.docsizes = docsizes if self.gaps_included else self.docntokens
        if self.gaps_included:
            self.docsizes[0] += style_knnlm.utils.DatasetTools.doc_skips_before(self.dataset, doc_slices=doc_slices)

    def __getitem__(self, index):
        style_indices = get_style_indices([index], self)[0]
        sample = self.dataset[index]
        sample['style'] = torch.from_numpy(self.style[style_indices])
        return sample

    def collater(self, samples): 
        """
        Adds style attribute entry to the mini-batch returned by ``MonolingualDataset.collater``.
        """

        collated = self.dataset.collater(samples) 
        style_values = [sample['style'] for sample in samples]
        collated['style'] = collate_style(style_values)
        return collated
        
    def __len__(self):
        return len(self.dataset)

    def _make_source_target(self, source, future_target, past_target):
        return self.dataset._make_source_target(source, future_target, past_target)

    def _maybe_add_bos(self, source, target):
        return self.dataset._maybe_add_bos(source, target)

    def _filter_vocab(self, target):
        return self.dataset._filter_vocab(target)

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)
    
    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

