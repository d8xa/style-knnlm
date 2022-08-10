import argparse
from pathlib import Path
import numpy as np
import faiss
import time
import logging

#from fairseq.progress_bar import tqdm_progress_bar
import style_knnlm.utils.functions
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dstore-mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--move-dstore-to-mem', default=False, action='store_true', help='load dstore to memory.')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--centroids', type=int, default=4096, dest="ncentroids", help='number of centroids faiss should learn')
parser.add_argument('--code-size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, dest="nprobe", help='number of clusters to query')
parser.add_argument('--filepath', type=str, help='file to write the faiss index to')
parser.add_argument('--batch-size-add', default=1000000, type=int, dest="batchsize", help='number of keys to add at a time (will be kept in memory)')
parser.add_argument('--batch-size-write', default=10000000, type=int, help='number of keys to add before writing the index to file')
parser.add_argument('--starting-point', type=int, help='index to start adding keys at')
parser.add_argument('--silent', default=False, action="store_true", help="supress status messages (the progress bar is exempt from this option)")
parser.add_argument('-log','--loglevel', default='warn', dest='loglevel')
args, _ = parser.parse_known_args()

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=args.loglevel.upper(),
)
logger = logging.getLogger('build_index')

def load_mmaps(logger, args):
    mmap_mode = 'r'
    paths = style_knnlm.utils.functions.mmap_paths(args)
    if args.move_dstore_to_mem:
        mmap_mode = None
        start = time.time()
        logger.info("Loading dstore to memory.")
    keys = np.load(paths["keys"], mmap_mode=mmap_mode)
    vals = np.load(paths["vals"], mmap_mode=mmap_mode)
    if args.move_dstore_to_mem:
        logger.info("Loading dstore to memory took {:.4g}s.".format(time.time() - start))
    return keys,vals

keys, vals = load_mmaps(logger, args)

def initialize():
    quantizer = faiss.IndexFlatL2(keys.shape[1])
    index = faiss.IndexIVFPQ(quantizer, keys.shape[1], args.ncentroids, args.code_size, 8)
    index.nprobe = args.nprobe
    return index

def train(index):
    logger.info('Starting training of index.')
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
    t = time.time()
    # Faiss does not handle adding keys in fp16 as of writing this.
    index.train(keys[random_sample].astype(np.float32))
    logger.info('Training index took {:.4g}s.'.format(time.time() - t))

def write(index, filename):
    logger.info('Writing to index {}.'.format(filename), end="")
    t = time.time()
    Path(args.filepath).parent.mkdir(exist_ok=True, parents=True)
    faiss.write_index(index, filename)
    logger.info('Writing to index took {:.4g}s.'.format(time.time() - t))

def populate(index):
    start = args.starting_point
    unsaved = 0
    max_unsaved = max(args.batch_size_write, args.batchsize) 

    with tqdm(total=vals.shape[0], initial=start, desc="Adding keys") as pbar:
        while start < vals.shape[0]:
            end = min(vals.shape[0], start+args.batchsize)
            batch = keys[start:end].copy()
            index.add_with_ids(batch.astype(np.float32), np.arange(start, end))
            start += batch.shape[0]
            unsaved += batch.shape[0]
            pbar.update(batch.shape[0])

            if unsaved >= max_unsaved:
                write(index, args.filepath)
                unsaved = 0

    if unsaved > 0:
        write(index, args.filepath)


if Path(args.filepath+".trained").exists():
    index = faiss.read_index(args.filepath+".trained")
else:
    index = initialize()
    train(index)
    write(index, args.filepath+".trained")

populate(index)