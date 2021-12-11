import argparse
import os
import numpy as np
import faiss
import time

#from fairseq.progress_bar import tqdm_progress_bar
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dstore-mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore-size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--key-dim', type=int, default=1024, dest="dim", help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--centroids', type=int, default=4096, dest="ncentroids", help='number of centroids faiss should learn')
parser.add_argument('--code-size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, dest="nprobe", help='number of clusters to query')
parser.add_argument('--filepath', type=str, help='file to write the faiss index to')
parser.add_argument('--batch-size-add', default=1000000, type=int, dest="batchsize",
                    help='number of keys to add at a time (will be kept in memory)')
parser.add_argument('--batch-size-write', default=10000000, type=int,
                    help='number of keys to add before writing the index to file')
parser.add_argument('--starting-point', type=int, help='index to start adding keys at')
parser.add_argument('--silent', default=False, action="store_true", 
                    help="supress status messages (the progress bar is exempt from this option)")
args = parser.parse_args()

print('\n'.join('{}={}'.format(k,v) for k, v in vars(args).items()))

dtypes = (np.float16, np.int16) if args.dstore_fp16 else (np.float32, np.int)

keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=dtypes[0], mode='r', shape=(args.dstore_size, args.dim))
vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=dtypes[1], mode='r', shape=(args.dstore_size, 1))

def initialize():
    quantizer = faiss.IndexFlatL2(args.dim)
    index = faiss.IndexIVFPQ(quantizer, args.dim, args.ncentroids, args.code_size, 8)
    index.nprobe = args.nprobe
    return index

def train(index):
    if not args.silent: print('Training index', end="")
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
    t = time.time()
    # Faiss does not handle adding keys in fp16 as of writing this.
    index.train(keys[random_sample].astype(np.float32))
    if not args.silent: print('; took {:.3f}s'.format(time.time() - t))

def write(index, filename):
    if not args.silent: print('Writing to index {}'.format(filename), end="")
    t = time.time()
    faiss.write_index(index, filename)
    if not args.silent: print('; took {:.3f}s'.format(time.time() - t))

def populate(index):
    start = args.starting_point
    unsaved = 0
    max_unsaved = max(args.batch_size_write, args.batchsize) 

    with tqdm(total=vals.shape[0], initial=start, desc="Adding keys") as pbar:
        while start < args.dstore_size:
            end = min(args.dstore_size, start+args.batchsize)
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


if os.path.exists(args.filepath+".trained"):
    index = faiss.read_index(args.filepath+".trained")
else:
    index = initialize()
    train(index)
    write(index, args.filepath+".trained")

populate(index)