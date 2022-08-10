import torch
import faiss
import logging
import numpy as np
from fairseq import utils
import time
from pathlib import Path

from .distance_metrics import dist_func
from style_knnlm import mmap_which, mmap_paths, mmap_shapes, mmap_dtypes
from style_knnlm.evaluation.flags import VariablesFlags as CacheFlags
from style_knnlm.caching import CachingManager

logger = logging.getLogger(__name__)

class KNN_Dstore(object):

    def __init__(self, args, empty=False, caching_mgr: CachingManager = None):
        self.half = args.dstore_fp16
        self.caching_mgr = caching_mgr
        if not empty:
            self.setup(args)

    def setup(self, args):
        """Load initialized datastore."""
        
        self.k = args.k
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.load_index(args)
        self.load_mmaps(args)
        self.last_query_result = None

    def load_index(self, args):
        """Load trained FAISS index for datastore."""

        start = time.time()
        self.index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info("Reading datastore took {:.2f}s.".format(time.time() - start))
        if not self.index.is_trained:
            raise ValueError("Index needs to be trained before it can be used.")
        self.index.nprobe = args.probe

    def make_mmaps(self, args, shape):
        """Initialize empty mmaps."""

        logger.info('Using {}-bit precision for dstore.'.format(
            16 if self.half else 32
        ))
        dtypes = mmap_dtypes(args)
        shapes = mmap_shapes(args, shape)
        paths = mmap_paths(args)
        which_mmaps = mmap_which(args)

        logger.info("Opening dstore with:\n{}.".format(
            '\n'.join([
                "{}: shape={}, dtype={}".format(k, shapes[k], dtypes[k])
                for k in which_mmaps
            ])
        ))
        if not Path(args.dstore_mmap).exists():
            Path(args.dstore_mmap).mkdir(exist_ok=True)
        
        for k in ['keys', 'vals', 'style']:
            if k in which_mmaps:
                setattr(self, k, np.lib.format.open_memmap(
                    paths[k], dtype=dtypes[k], shape=shapes[k], mode='w+')
                )
            else:
                setattr(self, k, None)
    
    def add_entry(self, index, hypo, args, style=None):
        """Save entry to mmaps."""

        tokens = hypo['tokens']
        keys = hypo['dstore_keys']
        length = keys.shape[0]
        if length == args.tokens_per_sample:
            if tokens.shape[0] != length:
                logger.info("value has length {}. Skipping entry.".format(tokens.shape[0]))
                return 0
            if index + length > self.keys.shape[0]:
                logger.info(
                    "Entry at index {} with length {} does not fit into remaining dstore"
                    " and will be truncated to length {}."
                    .format(index, length, self.keys.shape[0]-index)
                )
                length = self.keys.shape[0] - index
                keys = keys[:length]
                tokens = tokens[:length]
            self.keys[index:length+index] = (
                keys
                .view(-1, self.keys.shape[-1])
                .cpu().numpy()
                .astype(self.keys.dtype)
            )
            self.vals[index:length+index] = (
                tokens
                .view(-1, 1)
                .cpu().numpy()
                .astype(self.vals.dtype)
            )
            if self.style is not None:
                self.style[index:length+index] = (
                    style
                    [:length]
                    .view(-1, self.style.shape[-1])
                    .cpu().numpy()
                    .astype(self.style.dtype)
                )
        else:
            logger.info('Skipping entry with length {}.'.format(length))
            length = 0

        return length

    def load_mmaps(self, args):
        if not args.dstore_mmap:
            raise ValueError('Cannot build a datastore without the data.')

        dtypes = mmap_dtypes(args)
        paths = mmap_paths(args)

        mmap_mode = 'r'
        if args.move_dstore_to_mem:
            mmap_mode = None
            start = time.time()
            logger.info("Loading dstore to memory...")
        if not args.no_load_keys:
            self.keys = np.load(paths['keys'], mmap_mode=mmap_mode)
            assert self.keys.dtype == dtypes['keys']
        self.vals = np.load(paths['vals'], mmap_mode=mmap_mode)
        assert np.issubdtype(self.vals.dtype, dtypes['vals']), "dtype is {}, expected {}".format(self.vals.dtype, dtypes['vals'])
        if args.move_dstore_to_mem:
            logger.info("Loading to memory took {:.4g}s.".format(time.time() - start))

        if self.caching_mgr.all & CacheFlags.STYLE:
            self.style = np.load(paths['style'], mmap_mode=mmap_mode)

    def get_knns(self, queries):
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_knn_log_prob(self):
        if self.last_query_result is None:
            raise ValueError("No query results saved.")
        return self.last_query_result

    def run_query(self, queries, tgt, pad_idx, calc_vocab_prob=True, task=None):
        qshape = queries.shape # (T, B, C)
        queries = queries.view(-1, qshape[-1]) # (T * B, C)
        
        tgt = tgt.contiguous().view(-1)
        pad_mask = (tgt != pad_idx)
        reduced_tgt = tgt[pad_mask]

        dists, knns = self.get_knns(queries[pad_mask])
        knn_token_ids = self.vals[knns].squeeze(-1)

        dists = torch.from_numpy(dists).cuda() # (T_reduced * B, K)
        dists = dist_func(self, dists, knns, queries[pad_mask, :], function=self.sim_func)

        # caching
        if self.caching_mgr.all & CacheFlags.KNNS:
            self.caching_mgr.push(CacheFlags.KNNS, knns)
        if self.caching_mgr.all & CacheFlags.DISTS:
            self.caching_mgr.push(CacheFlags.DISTS, dists.cpu().numpy())
        if self.caching_mgr.all & CacheFlags.CORRECTNESS:
            self.caching_mgr.push(CacheFlags.CORRECTNESS, 
                self.get_correctness(knns, reduced_tgt, knn_token_ids=knn_token_ids)
            )
        if self.caching_mgr.all & CacheFlags.STYLE:
            self.caching_mgr.push(CacheFlags.STYLE, self.get_knn_style(knns))

        probs = utils.log_softmax(dists, dim=-1)

        knn_token_ids = torch.from_numpy(knn_token_ids).long().cuda()
        # to calculate only the prob on the ground truth tgt token for ppl
        index_mask = torch.eq(knn_token_ids, reduced_tgt.unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000  # for stability
        index_mask[index_mask == 1] = 0

        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone() # (T_reduced * B, K)
        full_yhat_knn_prob = torch.full([qshape[0] * qshape[1]], -10000.).cuda()
        full_yhat_knn_prob[pad_mask] = yhat_knn_prob

        if calc_vocab_prob:
            vocab_size = len(task.source_dictionary)
            full_yhat_knn_token_prob = self.get_vocab_probs(
                qshape, pad_mask, knn_token_ids, probs, vocab_size)

        self.last_query_result = (
            full_yhat_knn_prob.view(qshape[0], qshape[1], 1), # (T, B, 1)
            full_yhat_knn_token_prob.view(qshape[0], qshape[1], vocab_size) # (T, B, V)
            if calc_vocab_prob else None
        )

    def get_vocab_probs(self, qshape, pad_mask, knn_token_ids, probs, vocab_size):
        yhat_knn_token_prob = torch.full([knn_token_ids.shape[0], vocab_size], -10000.).cuda()
        for i, row in enumerate(knn_token_ids):
            unique_token_ids = row.unique()
            mask = torch.eq(knn_token_ids[i].repeat(
                unique_token_ids.shape[0], 1), unique_token_ids.unsqueeze(-1)).float()
            mask[mask == 0] = -10000
            mask[mask == 1] = 0
            yhat_knn_token_prob[i, unique_token_ids] = torch.logsumexp(
                probs[i].repeat(unique_token_ids.shape[0], 1) + mask, dim=-1).clone()
        full_yhat_knn_token_prob = torch.full([qshape[0] * qshape[1], vocab_size], -10000.).cuda()
        full_yhat_knn_token_prob[pad_mask] = yhat_knn_token_prob
        return full_yhat_knn_token_prob

    def maybe_get_knn_token_ids(self, knns, knn_token_ids=None):
        if knn_token_ids is None:
            return self.vals[knns].squeeze(-1)
        return knn_token_ids

    def get_correctness(self, knns, reduced_tgt, knn_token_ids=None):
        knn_token_ids = self.maybe_get_knn_token_ids(knns, knn_token_ids)
        correctness = (
            knn_token_ids == np.expand_dims(
                reduced_tgt.cpu().numpy(), 1).repeat(knns.shape[1], axis=1)
        ).astype(bool)
        return correctness

    def get_knn_style(self, knns):
        return self.style[knns]