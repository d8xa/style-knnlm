# k-Nearest-Neighbor Language Models with Style Attributes

This repository is a fork of the [knnlm](https://github.com/urvashik/knnlm) repository (see [commit](https://github.com/urvashik/knnlm/commits/8afab92bfcc8be28eccdf41fb82582a80977346e)) and adds support for style attributes.


## How to use

Preprocessing is exactly the same as `knnlm`.

LM training, LM evaluation, training of FAISS indices and kNN-LM evaluation have some changed or additional parameters.


### Data format for style attributes

Style attribute files follow the format of raw text files `{name}.{split}.tokens`: 
* name should be `{name}.{split}.style`
* values are separated by linebreaks, i.e. 1 value per line.
* 1 value for each line in the raw text.


### LM training

Training parameters for all previous models remain unchanged. For style attribute support we add the architecture `transformer_lm_style` as extension of `transformer_lm_wiki103`. 

Following the example in [knnlm](https://github.com/urvashik/knnlm):


```{bash}
python train.py \
    $BIN \
    --task language_modeling \
    --save-dir checkpoints/ \
    --arch transformer_lm_style \
    --style-input-dim 1 \
    --style-embed-dim 32 \
    --style-path $TEXT \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d
```

* `--style-input-dim`: the dimension of style attributes. Currently only `1` is supported.
* `--style-embed-dim`: the embedding dimension of style attributes in the LM.
* `--style-path`: the folder where style attribute files are saved.


### LM evaluation

```{bash}
python eval_lm.py \
    $BIN \
    --style-path $TEXT \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid
```


### Populating the datastore

```{bash}
python eval_lm.py \
    $BIN \
    --path checkpoints/checkpoint_best.pt \
    --style-path $TEXT \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/dstore --knn-keytype 'last_ffn_input' \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --dstore-save-style \
    --fp16 --dstore-fp16
```

* The `--dstore-size` parameter was removed, since we calculate the required size automatically and embed shape and dtype into the memmap.  
* We added an argument `--dstore-fp16` to enable saving in half precision.
* Use `--dstore-save-style` to save style attributes for the keys and values in the datastore. This is necessary for style metrics during evaluation.


### Training the FAISS index

The index training script `build_index.py` was rewritten and many parameters were renamed. Use the `--help` argument for more info.

```{bash}
python build_index.py \
    --dstore_mmap checkpoints/dstore \
    --filepath checkpoints/knn.index \
    --batch-size-add 1000000 \
    --batch-size-write 5000000 \
    --starting_point 0 \
    --move-dstore-to-mem \
    --dstore-fp16
```


### kNN-LM evaluation

```
python eval_lm.py \
    $BIN \
    --path checkpoints/checkpoint_best.pt \
    --style-path $TEXT \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-mmap checkpoints/dstore \
    --indexfile checkpoints/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --move-dstore-to-mem
```


#### Custom metrics

We added some custom evaluation metrics to `eval_lm.py`, which can be enabled separately with `--report-metrics`, and reported during/after evaluation. 

* `style`: Style MAE/MBE. The mean absolute/bias error of retrieved style vs. requested style.
* `topk-ds-precision`: Top-k datastore retrieval precision. Requires the additional parameter `--top-k` to be set.
* Top-k LM precision of probabilities. If `--knnlm` is used, all three probabilites will be used (LM, datastore, interpolated). Otherwise only LM probabilities will be used. Requires the additional parameter `--top-k` to be set. 



#### Saving intermediate variables

To support saving intermediate variables we adapt some of [efficient-knnlm](https://github.com/jxhe/efficient-knnlm)'s code.

Variables can be saved to `--save-vars-dir` with `--save-vars`. Options for `--save-vars` are:

* `predictions`: The top-k predictions (requires `--top-k`).  
If `--knnlm` is used, this includes predictions from LM,- datastore- and interpolated probabilities. Otherwise only from LM probabilities.
* `dictionary`: the vocabulary used
* `knns`: the indices of retrieved kNNs
* `dists`: the distances of the retrieved kNNs to the queries
* `reftargets`: the reference targets
* `refstyle`: the reference style
* `style`: the retrieved style
* `correctness`: Boolean mask where the retrieved kNNs match the reference target token.


## Known issues

* Calculating/saving top-k probabilities fails with large models due to CUDA OOM errors.   
