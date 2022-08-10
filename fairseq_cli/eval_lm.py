#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import LMContextWindowDataset, data_utils
from fairseq.data.lm_context_window_style_dataset import LMContextWindowStyleDataset
from fairseq.data.monolingual_style_dataset import MonolingualStyleDataset
from fairseq.dstore.dstore import KNN_Dstore
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from style_knnlm.caching import CachingManager
from style_knnlm.evaluation import metrics
from style_knnlm.evaluation.flags import (
    CustomMetricsFlags as MFlags,
    VariablesFlags as VFlags
)

logger = logging.getLogger(__name__)


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(
            self.word, self.count, self.log_prob, self.is_bpe, 
            self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    task, args, models, dataset, itr = Helpers.setup_eval(parsed_args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    max_positions = utils.resolve_max_positions(*[model.max_positions() for model in models])

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch, args=args)
    score_meter = AverageMeter()
    bpe_toks, bpe_len = process_bpe(task, args)
    word_stats = dict()

    cache_flags = VFlags.NONE
    metrics_flags = MFlags.NONE
    if not args.save_knnlm_dstore:
        metrics_flags = set_metrics_flags(args) # which metrics to meter/report
        cache_flags = set_caching_flags(args, metrics_flags)
        write_flags = set_saving_flags(args)
        cache_flags |= write_flags
        validate_flags(args, metrics_flags, cache_flags)

        caching = CachingManager(flags=cache_flags, persist=write_flags)
        custom_meters = initialize_custom_meters(metrics_flags, cache_flags)

    if args.knnlm:
        if args.save_knnlm_dstore:
            raise ValueError("Cannot use knnlm while trying to build the datastore!")
        knn_dstore = KNN_Dstore(args, caching_mgr=caching)
    elif args.save_knnlm_dstore:
        knn_dstore = KNN_Dstore(args, empty=True)
        dstore_shape = calc_dstore_shape(args, dataset, max_positions)
        knn_dstore.make_mmaps(args, dstore_shape)
        dstore_idx = 0
 
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        for ex_i, sample in enumerate(t):
            if 'net_input' not in sample:
                continue

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if not args.save_knnlm_dstore:
                cache_sample(caching, sample)

            gen_timer.start()
            if args.knnlm:
                hypos = scorer.generate(models, sample, knn_dstore=knn_dstore, task=task, flags=cache_flags)
            else:
                hypos = scorer.generate(models, sample, flags=cache_flags)
            gen_timer.stop(sample['ntokens'])

            if MFlags.TOPK_DS_PRECISION in metrics_flags:
                custom_meters[MFlags.TOPK_DS_PRECISION].update(
                    metrics.topk_knn_precision(
                        caching.peek(VFlags.CORRECTNESS), k=args.top_k
                    )
                )
            
            bsz = sample['target'].shape[0]
            offsets = sample['start_indices'] if 'start_indices' in sample else [0] * bsz

            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]

                if args.save_knnlm_dstore:
                    style = truncate_style(sample, i) if 'style' in sample else None
                    dstore_idx += knn_dstore.add_entry(dstore_idx, hypo, args, style=style)
                    continue

                sample_id = sample['id'][i]
                tokens = hypo['tokens']
                tgt_len = tokens.numel()
                pos_scores = hypo['positional_scores'].float()
                
                if args.add_bos_token:
                    assert hypo['tokens'][0].item() == task.target_dictionary.bos()
                    tokens = tokens[1:]
                    pos_scores = pos_scores[1:]

                skipped_toks = skip_bpe_scores(bpe_toks, tokens, tgt_len, pos_scores)
                pos_scores = skip_inf_scores(task, tokens, pos_scores) 

                if not args.save_knnlm_dstore:
                    cache_hypo(caching, sample, hypo, i)

                    update_custom_meters_inner(args, caching, custom_meters, i, tgt_len, offsets)
                    val = pos_scores.sum().cpu()
                    n = pos_scores.numel() - skipped_toks
                    score_meter.update(val=val/n, n=n)

                if args.output_word_probs or args.output_word_stats:
                    word_prob = get_word_prob(task, bpe_toks, bpe_len, word_stats, tokens, pos_scores)
                    if args.output_word_probs:
                        logger.info(
                            str(int(sample_id)) + " "
                            + ('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob))
                        )

            wps_meter.update(sample['ntokens'])
            pbar_metrics = {'wps': round(wps_meter.avg)}
            if not args.save_knnlm_dstore:
                pbar_metrics.update(summary(score_meter, custom_meters))
            t.log({k:f'{v:.4g}' for k,v in pbar_metrics.items() if not np.isinf(v) or np.isna(v)})

    if args.save_knnlm_dstore:
        logger.info("Saved {} tokens to datastore {:.1f}s ({:.2f} tokens/s)."
        .format(dstore_idx, gen_timer.sum, 1. / gen_timer.avg))
        return

    logger.info("Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)"
        .format(gen_timer.n, gen_timer.sum, 1. / gen_timer.avg)
    )
    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)

    if write_flags:
        write_timer = StopwatchMeter()
        write_timer.start()
        logger.info("Writing cached variables to disk...")
        if args.save_vars_dir is None:
            if getattr(args, "dstore_mmap", None) is not None and args.knnlm:
                cache_savepath = Path(args.dstore_mmap).joinpath("cache", args.gen_subset)
            else:
                raise ValueError("Savepath for caches needs to be specified with --save-vars-dir")
        cache_savepath = Path(args.save_vars_dir)
        cache_savepath.mkdir(exist_ok=True, parents=True)

        if 'dictionary' in args.save_vars:
            task.dictionary.save(str(cache_savepath.joinpath("dictionary.txt")))
        caching.write(cache_savepath)
        write_timer.stop()
        logger.info("Writing cached variables took {:.4g}s.".format(write_timer.elapsed_time))
    
    # report final state of custom metrics
    metrics_report = summary(score_meter, custom_meters)
    logger.info('Metrics report:\n' + '\n'.join(
        [f"{k}: {v:.4g}" for k,v in metrics_report.items()]
    ))

    return metrics_report

def update_custom_meters_inner(args, caching, custom_meters, i, tgt_len, offsets):
    for flag in [MFlags.STYLE_MBE, MFlags.STYLE_MAE]:
        if flag in custom_meters:
            custom_meters[flag].update(
                metrics.topk_style_error(
                    caching.peek(VFlags.STYLE_REFERENCE),
                    caching.peek(VFlags.STYLE)[i],
                    k=1, absolute=flag & MFlags.STYLE_MAE
                )
            )
    for flag in [VFlags.PRED_LM_TOPK, VFlags.PRED_DS_TOPK, VFlags.PRED_INTERP_TOPK]:
        if flag in custom_meters:
            custom_meters[flag].update(
                metrics.topk_vocab_precision(
                    caching.peek(VFlags.TARGET_REFERENCE)[i][offsets[i]:offsets[i]+tgt_len],
                    caching.peek(flag),
                    k=args.top_k
                )
            )

def summary(score_meter, other_meters: dict):
    avg_nll_loss = - score_meter.avg / math.log(2)  # convert to base 2
    output = {
        flag.name.lower(): meter.agg
        for flag,meter in other_meters.items()
        if getattr(meter, 'n', 0) > 0 or getattr(meter, 'count', 0) > 0
    }
    output.update({
        'loss (base 2)': avg_nll_loss.item(), 
        'ppl': (2**avg_nll_loss).item()
    })
    return output


def cache_hypo(caching_mgr: CachingManager, sample, hypo, i):
    """Cache all variables on hypothesis level."""

    for flag in [VFlags.PRED_LM_TOPK, VFlags.PRED_DS_TOPK, VFlags.PRED_INTERP_TOPK]:
        if flag in hypo:
            caching_mgr.push(flag, hypo[flag])
    if VFlags.STYLE_REFERENCE in caching_mgr.all:
        caching_mgr.push(VFlags.STYLE_REFERENCE, truncate_style(sample, i).detach().cpu())


def truncate_style(sample, i):
    s = sample['start_indices'][i]
    if s == 0 and len(sample['style'][i]) > len(sample['target'][i]):
        s = len(sample['style'][i]) - len(sample['target'][i])
    return sample['style'][i][s:]


def cache_sample(caching_mgr: CachingManager, sample):
    """Cache all variables on sample level."""

    if VFlags.TARGET_REFERENCE & caching_mgr.all:
        caching_mgr.push(VFlags.TARGET_REFERENCE, 
            sample['target'] # (B, T, C)
            #.permute(1,0) # top-k indices have shape (B, T, k)
            .detach()
            .cpu()
        )


def initialize_custom_meters(metrics_flags, cache_flags):
        custom_meters = {}
        for flag in [MFlags.STYLE_MAE, MFlags.STYLE_MBE, MFlags.TOPK_DS_PRECISION]:
            if flag & metrics_flags:
                custom_meters[flag] = AverageMeter()
        if MFlags.TOPK_VOCAB_PRECISION & metrics_flags:
            custom_meters[VFlags.PRED_LM_TOPK] = AverageMeter()
            if cache_flags & VFlags.PRED_DS_TOPK:
                custom_meters[VFlags.PRED_DS_TOPK] = AverageMeter()
            if cache_flags & VFlags.PRED_INTERP_TOPK:
                custom_meters[VFlags.PRED_INTERP_TOPK] = AverageMeter()
        if MFlags.TOPK_DS_PRECISION & metrics_flags:
            custom_meters[MFlags.TOPK_DS_PRECISION] = AverageMeter()
        return custom_meters

def validate_flags(args, metrics_flags: MFlags, caching_flags: VFlags):
    """Validate flags with current configuration."""
    pass
    if MFlags.STYLE in metrics_flags:
        if not args.knnlm:
            raise ValueError("Can't evaluate retrieved style in non-knnlm mode.")
        if not args.style_path:
            raise ValueError("Can't evaluate style without reference attributes.")
        # TODO: check if dstore has style
        #     raise ValueError("Can't evaluate retrieved style"
        #         " without saving style to dstore.")
    if MFlags.TOPK_VOCAB_PRECISION in metrics_flags or MFlags.TOPK_DS_PRECISION in metrics_flags:
        if args.top_k is None:
            raise ValueError("Can't calculate top-k metrics without k. Use --top-k to set k.")

def set_metrics_flags(args):
    """Set flags for requested metrics."""
    flags = MFlags.NONE
    for x in args.report_metrics:
        if x == 'style':
            flags |= MFlags.STYLE
        elif x == 'topk-vocab-precision':
            flags |= MFlags.TOPK_VOCAB_PRECISION
        elif x == 'topk-ds-precision':
            flags |= MFlags.TOPK_DS_PRECISION
    return flags

def set_caching_flags(args, metrics_flags: MFlags):
    """Set flags for caching of variables necessary for requested metrics."""
    flags = VFlags.NONE
    if metrics_flags:
        if MFlags.STYLE & metrics_flags:
            flags |= VFlags.STYLE_REFERENCE | VFlags.STYLE
        if MFlags.TOPK_VOCAB_PRECISION & metrics_flags:
            flags |= VFlags.TARGET_REFERENCE
            flags |= VFlags.PRED_LM_TOPK
            if args.knnlm:
                flags |= VFlags.PRED_DS_TOPK | VFlags.PRED_INTERP_TOPK
        if MFlags.TOPK_DS_PRECISION:
            flags |= VFlags.CORRECTNESS
    return flags

def set_saving_flags(args):
    """Set flags for variables to save after evaluation."""
    flags = VFlags.NONE
    if not args.save_vars:
        return flags
    flag_arg_map = {
        'refstyle': VFlags.STYLE_REFERENCE, 'style': VFlags.STYLE,
        'knns': VFlags.KNNS, 'dists': VFlags.DISTS, 
        'reftargets': VFlags.TARGET_REFERENCE, 
        'correctness': VFlags.CORRECTNESS
    }
    for name,flag in flag_arg_map.items():
        if name in args.save_vars:
            flags |= flag
    if 'predictions' in args.save_vars:
        flags |= VFlags.PRED_LM_TOPK
        if args.knnlm:
            flags |= VFlags.PRED_DS_TOPK | VFlags.PRED_INTERP_TOPK
    return flags

def skip_bpe_scores(bpe_toks, tokens, tgt_len, pos_scores):
    skipped_toks = 0
    if bpe_toks is not None:
        for i in range(tgt_len - 1):
            if tokens[i].item() in bpe_toks:
                skipped_toks += 1
                pos_scores[i + 1] += pos_scores[i]
                pos_scores[i] = 0
    return skipped_toks

def skip_inf_scores(task, tokens, pos_scores):
    inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
    if inf_scores.any():
        logger.debug(
            'skipping tokens with inf scores: {}'.format(
                task.target_dictionary.string(tokens[inf_scores.nonzero()])
            )
        )
        pos_scores = pos_scores[(~inf_scores).nonzero()]
    return pos_scores

def get_word_prob(task, bpe_toks, bpe_len, word_stats, tokens, pos_scores):
    w = ''
    word_prob = []
    is_bpe = False
    for i in range(len(tokens)):
        w_ind = tokens[i].item()
        w += task.source_dictionary[w_ind]
        if bpe_toks is not None and w_ind in bpe_toks:
            w = w[:-bpe_len]
            is_bpe = True
        else:
            word_prob.append((w, pos_scores[i].item()))

            next_prob = None
            ind = i + 1
            while ind < len(tokens):
                if pos_scores[ind].item() != 0:
                    next_prob = pos_scores[ind]
                    break
                ind += 1

            word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
            is_bpe = False
            w = ''
    return word_prob

def process_bpe(task, args):
    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    return bpe_toks,bpe_len

def calc_dstore_shape(args, dataset, max_positions):
    # only count tokens at indices that are actually used by batch iterator
    indices = data_utils.filter_by_size(dataset.ordered_indices(), dataset, max_positions)
    dstore_tokens = dataset.sizes[indices].sum()
    
    shape = (
        dstore_tokens - dataset.context_window, 
        args.decoder_embed_dim + getattr(args, "style_embed_dim", 0)
    )
    return shape

class Helpers:
    def setup_eval(parsed_args):
        task = Helpers.setup_task(parsed_args)
        task, args, models = Helpers.load_ensemble(task, parsed_args)
        dataset = Helpers.load_dataset(task, args)
        Helpers.optimize_ensemble(models, args)
        itr = Helpers.get_batch_iterator(task, args, models, dataset)
        return task, args, models, dataset, itr

    def setup_task(parsed_args):
        utils.import_user_module(parsed_args)
        task = tasks.setup_task(parsed_args)
        return task

    def load_ensemble(task, parsed_args):
        logger.info('loading model(s) from {}'.format(parsed_args.path))
        models, args = checkpoint_utils.load_model_ensemble(
            parsed_args.path.split(os.pathsep),
            arg_overrides=eval(parsed_args.model_overrides),
            task=task,
        )

        for arg in vars(parsed_args).keys():
            if arg not in {
                'self_target', 'future_target', 'past_target', 'tokens_per_sample',
                'output_size_dictionary', 'add_bos_token',
            }:
                setattr(args, arg, getattr(parsed_args, arg))

        # reduce tokens per sample by the required context window size
        if args.context_window > args.tokens_per_sample:
            raise ValueError("Context window larger than sample size not supported.")
        logger.debug("Reducing tokens_per_sample={} by context_window={}. New tokens_per_sample={}".format(
            args.tokens_per_sample, args.context_window, args.tokens_per_sample - args.context_window
        ))
        args.tokens_per_sample -= args.context_window
        task = tasks.setup_task(args)

        return task, args, models

    def load_dataset(task, args):
        task.load_dataset(args.gen_subset)
        dataset = task.dataset(args.gen_subset)

        if args.context_window > 0:
            if type(dataset) is MonolingualStyleDataset:
                dataset = LMContextWindowStyleDataset(
                    dataset=dataset,
                    tokens_per_sample=args.tokens_per_sample,
                    context_window=args.context_window,
                    pad_idx=task.source_dictionary.pad(),
                )
            else:
                dataset = LMContextWindowDataset(
                    dataset=dataset,
                    tokens_per_sample=args.tokens_per_sample,
                    context_window=args.context_window,
                    pad_idx=task.source_dictionary.pad(),
                )

        logger.info("Loaded {} examples from {} (subset \'{}\') as {}.".format(
            len(dataset), args.data, args.gen_subset, type(dataset).__name__
        ))
        return dataset

    def optimize_ensemble(models, args, use_cuda=None):
        """Optimize ensemble for generation and set the source 
        and dest dicts on the model (required by scorer)"""
        if use_cuda is None:
            use_cuda = torch.cuda.is_available() and not args.cpu
        for model in models:
            model.make_generation_fast_()
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()
        assert len(models) > 0
        logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    def get_batch_iterator(task, args, models, dataset, shuffle=False):
        return task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens or 36000,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in models
            ]),
            ignore_invalid_inputs=True,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=shuffle)


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    result = main(args)
    print(result)


if __name__ == '__main__':
    cli_main()
