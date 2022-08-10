from pathlib import Path
import copy
import logging

from fairseq.data.token_block_utils_fast import (
    _get_slice_indices_fast
    , _get_block_to_dataset_index_fast
)
import numpy as np
import torch as tc
from .layered import list_layers, pick_layer

logger = logging.getLogger(__name__)

def as_numpy_dtype(dtype):
    dtype_dict = {
        tc.bool    : np.dtype(np.bool),
        tc.uint8   : np.dtype(np.uint8),
        tc.int8    : np.dtype(np.int8),
        tc.int16   : np.dtype(np.int16),
        tc.short   : np.dtype(np.int16),
        tc.int32   : np.dtype(np.int32),
        tc.int     : np.dtype(np.int32),
        tc.int64   : np.dtype(np.int64),
        tc.long    : np.dtype(np.int64),
        tc.float16 : np.dtype(np.float16),
        tc.half    : np.dtype(np.float16),
        tc.float32 : np.dtype(np.float32),
        tc.float   : np.dtype(np.float32),
        tc.float64 : np.dtype(np.float64),
        tc.double  : np.dtype(np.float64),
    }
    return dtype_dict[dtype]

#def as_torch_dtype(dtype: np.dtype)
# TODO

def load_style_attributes(path):
    if "csv" in Path(path).suffix:
        try:
            values = np.loadtxt(path, dtype=float)
        except ValueError: # assume that error is due to header in csv
            values = np.loadtxt(path, dtype=float, skiprows=1)
    else:
        values = np.loadtxt(path, dtype=float)
    if values.ndim==1:
        values = values.reshape(-1,1)
    logger.info(f"Read {len(values)} style attributes from {path}.")
    return values


def mmap_which(args):
        l = ["keys", "vals"]
        if (
            'style' in getattr(args, 'report_metrics', []) 
            or 'style' in getattr(args, 'save_vars', []) 
            or getattr(args, 'dstore_save_style', False)
        ):
            return l + ["style"]
        return l

def mmap_dtypes(args):
    fp16 = getattr(args, "dstore_fp16", False)
    d = {
        "keys": np.dtype(np.float16 if fp16 else np.float32),
        "vals": np.dtype(np.int16 if fp16 else np.int32),
        # NOTE: previously `int` was used, but this could lead to compatibility issues, since numpy chooses `np.int32` or `np.int64` depending on platform. 
        # Since the vocabulary size << 2**32, `np.int64` won't be needed.
        "style": np.dtype(np.float16 if fp16 else np.float32)
        # TODO: decide how to handle integer/boolean style attributes.
    }
    return {k: d[k] for k in mmap_which(args)}

def mmap_shapes(args, shape):
    d = {
        "keys": shape,
        "vals": (shape[0], 1),
        "style": (shape[0], getattr(args, "style_input_dim", 0))
    }
    return {k: d[k] for k in mmap_which(args)}

def mmap_paths(args):
    return {k: str(Path(args.dstore_mmap).joinpath(f"{k}.npy")) for k in mmap_which(args)}


class DatasetTools:
    def base_dataset(dataset):
        """Returns the innermost dataset from a nested dataset."""
        return list_layers(dataset)[-1]

    def token_counts(dataset):
        """Computes sample sizes fo the base dataset."""
        return DatasetTools.base_dataset(dataset).sizes

    def doc_slices(dataset, token_counts=None):
        """Computes document index slices of documents in dataset."""
        if token_counts is None:
            token_counts = DatasetTools.token_counts(dataset).astype(np.int64) # (L,)
        doc_slices = _get_slice_indices_fast(token_counts.astype(np.int64), "complete_doc", 0, 1) # (D,2)
        return doc_slices

    def doc_sizes(dataset, doc_slices=None):
        """Computes document sizes of documents in dataset."""
        if doc_slices is None:
            doc_slices = DatasetTools.doc_slices(dataset)
        doc_sizes = np.diff(doc_slices).ravel() # (D,)
        return doc_sizes

    def doc_skips(dataset, doc_slices=None, token_counts=None):
        """Calculates the numer of skipped tokens in the dataset."""
        if doc_slices is None:
            doc_slices = DatasetTools.doc_slices(dataset)
        if token_counts is None:
            token_counts = DatasetTools.token_counts(dataset)
        toks_total = token_counts.sum()
        skips_before = DatasetTools.doc_skips_before(dataset, doc_slices)
        skips_after = DatasetTools.doc_skips_after(dataset, doc_slices, toks_total)

        skips = np.concatenate([
            doc_slices[1:,0] - doc_slices[:-1,1], # between samples
            [skips_after] # after last sample
        ])
        skips[0] += skips_before
            
        return skips

    def doc_skips_before(dataset, doc_slices=None):
        """Calculates the number of skipped tokens before the first sample."""
        if doc_slices is None:
            return DatasetTools.doc_slices(dataset)[0,0]
        return doc_slices[0,0]

    def doc_skips_after(dataset, doc_slices=None, toks_total=None):
        """Calculates the number of skipped tokens after the first sample."""
        if doc_slices is None:
            doc_slices = DatasetTools.doc_slices(dataset)
        if toks_total is None:
            toks_total = DatasetTools.token_counts(dataset).sum()
        if doc_slices[-1,1] < toks_total:
            return toks_total - doc_slices[-1,1]
        return 0

    def block_slices(dataset):
        """Returns TokenBlock slice indices."""
        from fairseq.data.token_block_dataset import TokenBlockDataset

        return pick_layer(dataset, TokenBlockDataset).slice_indices


def sample_to_token_slices(dataset, slices):
    """
    Converts context expanded sample index slices to token index slices.

    Parameters
    ----------
    dataset : FairseqDataset
    slices : np.ndarray
        A (B,3)-dim array containing sample-to-document-index slices.
    """
    from fairseq.data.monolingual_dataset import MonolingualDataset

    sz = np.concatenate([[0], pick_layer(dataset, MonolingualDataset).sizes.cumsum()])
    token_blocks = np.array([[sz[a]+offset, sz[b+1]] for a,offset,b in slices])

    return token_blocks

def tokens_per_document(index, dataset, docslices, docsizes):
    """
    Calculates tokens per document for all documents in a single sample at index.

    Parameters
    ----------
    index : int
        The index in the dataset.
    dataset : FairseqDataset
        A dataset containing B samples.
    docslices : np.ndarray
        A (B,3)-dim array of slices containing document indices and token start offsets for each sample.
    docsizes : np.ndarray
        A D-dim array containing document sizes (token counts).
    """

    current_slice = docslices[index]
    offsets = [current_slice[1], 0]
    if index+1 < len(dataset):
        next_slice = docslices[index+1]
        if next_slice[1] > 0:
            offsets[1] = next_slice[1] # number of tokens from last document in sample, if cut off.
    document_sizes = copy.deepcopy(docsizes[current_slice[0]:current_slice[-1]+1])
    if offsets[1] > 0:
        document_sizes[-1] = offsets[1] # replace last token count by remaining, if cut off.
    document_sizes[0] -= offsets[0] # reduce first document token count by starting offset.
    document_indices = np.arange(current_slice[0], current_slice[-1]+1)

    return document_indices, document_sizes

def context_expansion(index, capacity, sizes):
    """Expands sample indices with indices of leftward context.
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

def initialize_index_arrays(dataset):
    """Sets slice index arrays and size arrays necessary for context expansion of samples."""

    from fairseq.data.token_block_dataset import TokenBlockDataset

    dataset.sample_lens = pick_layer(dataset, TokenBlockDataset).sizes
    max_sample_len = dataset.tokens_per_sample + dataset.context_window
    capacity = np.minimum(max_sample_len - dataset.sample_lens, dataset.context_window)
    dataset.slice_indices_sample = np.array([
        context_expansion(i, capacity[i], dataset.sample_lens) for i in range(len(dataset))
    ])