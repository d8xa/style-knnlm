# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
import numpy as np

from fairseq import utils
from style_knnlm.evaluation.flags import VariablesFlags
from fairseq.dstore.dstore import combine_knn_and_vocab_probs

import logging
logger = logging.getLogger(__name__)


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, args=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']
        style = sample.get('style', None)
        if self.args.fp16 and style is not None:
            style = style.half()

        flags: VariablesFlags = kwargs['flags']
        topk = 0
        if VariablesFlags.PRED_LM_TOPK in flags:
            topk = self.args.top_k
        
        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
            if coeff == 1:
                return knn_p
            elif coeff == 0:
                return vocab_p
            combine_probs = torch.stack([vocab_p, knn_p], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - coeff)
            coeffs[1] = np.log(coeff)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

            return curr_prob

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input, style=style)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            full_probs = None
            for i, (bd, tgt, is_single) in enumerate(batched):
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data

                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                    full_probs = curr_prob if topk > 0 else None
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    if full_probs is None and topk > 0:
                        full_probs = curr_prob.new(
                            torch.Size([curr_prob.size(0), orig_target.numel(), curr_prob.size(2)]))
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    if topk > 0:
                        full_probs[:, idx:end, :] = curr_prob
                        full_probs[:, idx:end, :][curr_prob == 0.] = -1e-4
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)
            if full_probs is not None:
                full_probs[full_probs == 0.] = -1e4
                full_probs = full_probs.squeeze(0).view(sample['target'].shape[0], sample['target'].shape[1], -1)

            if 'knn_dstore' in kwargs:
                dstore = kwargs['knn_dstore']
                # TxBxC
                queries = bd[1][self.args.knn_keytype]
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                dstore.run_query(
                    queries=queries,
                    tgt=orig_target.permute(1, 0),
                    pad_idx=self.pad,
                    task=kwargs["task"],
                    calc_vocab_prob=(flags.PRED_LM_TOPK | flags.PRED_DS_TOPK | flags.PRED_INTERP_TOPK) & flags
                )
                yhat_knn_prob, yhat_knn_vocab_prob = dstore.get_knn_log_prob()
                yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
                if yhat_knn_vocab_prob is not None:
                    yhat_knn_vocab_prob = yhat_knn_vocab_prob.permute(1, 0, 2)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    if yhat_knn_vocab_prob is not None:
                        yhat_knn_vocab_prob = yhat_knn_vocab_prob.half()
                    probs = probs.half()

                probs = combine_knn_and_vocab_probs(
                    yhat_knn_prob, probs, self.args.lmbda)

                if full_probs is not None:
                    actual_dtype = full_probs.dtype
                    if topk > 0:
                        topk_indices_lm = torch.topk(full_probs.float(), k=topk, dim=-1)[-1].detach().cpu().to(actual_dtype)
                        topk_indices_knn = torch.topk(yhat_knn_vocab_prob.float(), k=topk, dim=-1)[-1].detach().cpu().to(actual_dtype)
                    full_probs = combine_knn_and_vocab_probs(
                        yhat_knn_vocab_prob, full_probs, self.args.lmbda, cpu=True)

                    if topk > 0:
                        topk_indices = torch.topk(full_probs.float(), k=topk, dim=-1)[-1].detach().cpu().to(actual_dtype)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz

        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad)
                if sample['target'] is not None 
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None

            hypo = {
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'dstore_keys': None
            }
            
            if topk > 0:
                if flags & flags.PRED_LM_TOPK:
                    hypo[flags.PRED_LM_TOPK] = take_topk(topk_indices_lm, i, start_idxs, tgt_len)
                if flags & flags.PRED_DS_TOPK:
                    hypo[flags.PRED_DS_TOPK] = take_topk(topk_indices_knn, i, start_idxs, tgt_len)
                if flags & flags.PRED_INTERP_TOPK:
                    hypo[flags.PRED_INTERP_TOPK] = take_topk(topk_indices, i, start_idxs, tgt_len)
        
            if self.args.save_knnlm_dstore:
                hypo['dstore_keys'] = decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:]

            hypos.append([hypo])
        return hypos


def take_topk(indices, i, offsets, tgt_len):
    return indices[i][offsets[i]:offsets[i] + tgt_len].cpu()#.numpy()