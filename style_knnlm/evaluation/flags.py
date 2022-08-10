from enum import Flag, auto


class CustomMetricsFlags(Flag):
    """Selection of (metrics|values) to (meter|report) (during|after) model evaluation."""

    NONE = 0
    
    STYLE_MAE = auto()
    STYLE_MBE = auto()
    STYLE = STYLE_MAE | STYLE_MBE
    TOPK_VOCAB_PRECISION = auto() # precision across top-k vocab tokens (LM/DS/LM+DS)
    TOPK_DS_PRECISION = auto() # precision across top-k retrieved kNNs


class VariablesFlags(Flag):
    """Selection of variables to (cache|save) (during|after) model evaluation."""

    NONE = 0

    #TARGET = auto() # predicted target REVIEW: removed bc. topk can be used instead
    TARGET_REFERENCE = auto() # reference target

    STYLE = auto() # style retrieved from DS (for all kNNs)
    STYLE_REFERENCE = auto() # reference style

    PRED_LM_TOPK = auto() # top-k predicted tokens by LM probs
    PRED_DS_TOPK = auto() # ... by DS probs
    PRED_INTERP_TOPK = auto() # ... by LM+DS probs

    KNNS = auto() # DS indices of kNNs
    DISTS = auto() # distances of kNNs to the query vector
    CORRECTNESS = auto() # boolean mask where retrieved targets match the reference target.