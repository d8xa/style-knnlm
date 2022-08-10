import numpy as np
import torch as tc


def topk_knn_precision(correctness: np.ndarray, k: int = None):
    """Computes the top-k precision on kNNs"""
    return correctness[:,:k].mean()

def topk_style_error(ref_style: tc.tensor, style: np.ndarray, k: int = None, absolute=False):
    """Computes the MBE or MAE on kNN-retrieved style."""
    if style.ndim==2:
        devs = np.expand_dims(ref_style.cpu().numpy(), axis=1) - style[:k,:]
    elif style.ndim==3:
        devs = np.expand_dims(ref_style.cpu().numpy(), axis=1) - style[:,:k,:]
    if absolute:
        return np.mean(np.abs(devs))
    return np.mean(devs)

def topk_vocab_precision(refs: tc.tensor, topk_vocab: np.ndarray, k: int = None):
    """Computes precision on the top-k vocabulary tokens (top-k by probability)."""
    return (
        # the first sample in the dataset has empty context and therefore no predictions, so
        # those reference targets are skipped
        (refs[-len(topk_vocab):].reshape(-1, 1) == topk_vocab[:,:k])
        .any(dim=1)
        .to(float)
        .mean()
        .item()
    )