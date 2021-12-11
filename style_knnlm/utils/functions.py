from pathlib import Path
import copy

from fairseq.data.token_block_utils_fast import (
    _get_slice_indices_fast
    , _get_block_to_dataset_index_fast
)
import numpy as np
from fairseq.data.token_block_dataset import TokenBlockDataset
from fairseq.data.indexed_dataset import MMapIndexedDataset
from .layered import pick_layer

def load_style_attributes(path):
    with Path(path).open("r") as f:
        return [float(x) for x in f.read().split("\n\n")]

def get_doc_slices(dataset):
    token_counts = pick_layer(dataset, MMapIndexedDataset).sizes.astype(np.int64) # (L,)
    doc_slices = _get_slice_indices_fast(token_counts, "complete_doc", 0, 1) # (D,2)
    return doc_slices

def get_doc_sizes(dataset):
    doc_slices = get_doc_slices(dataset)
    doc_sizes = np.diff(doc_slices).ravel() # (D,)
    return doc_sizes

def get_block_slices(dataset):
    return pick_layer(dataset, TokenBlockDataset).slice_indices

def get_doc_gaps(dataset):
    doc_slices = get_doc_slices(dataset)
    gaps = np.concatenate([doc_slices[1:,0] - doc_slices[:-1,1], [0]])
    return gaps

def sample_to_token_slices(dataset, slices):
    """
    Converts context expanded sample index slices to token index slices.

    Parameters
    ----------
    dataset : FairseqDataset
    slices : np.ndarray
        A (B,3)-dim array containing sample-to-document-index slices.
    """

    sz = np.concatenate([[0], dataset.sizes.cumsum()])
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

def sample_to_document_slices(dataset, gaps=True):
    """Map document slices to samples."""

    # L: nr of separated documents
    # D: nr of documents
    # B: nr of samples
    token_counts = pick_layer(dataset, MMapIndexedDataset).sizes.astype(np.int64) # (L,)
    doc_slices = _get_slice_indices_fast(token_counts, "complete_doc", 0, 1) # (D,2)
    doc_sizes = np.diff(doc_slices).ravel() # (D,)
    if gaps:
        doc_sizes += np.concatenate([doc_slices[1:,0] - doc_slices[:-1,1], [0]]) # (D,)
    block_indices = pick_layer(dataset, TokenBlockDataset).slice_indices # (B,2)
    I = _get_block_to_dataset_index_fast(doc_sizes, block_indices) # (B, 3)

    return I