import os, sys, torch, fairseq

def build_command(command_strings):
    sys.argv = command_strings
    parser = fairseq.options.get_eval_lm_parser()
    parsed_args = fairseq.options.parse_args_and_arch(parser)
    
    return parsed_args

def setup_task(args):
    fairseq.utils.import_user_module(args)
    task = fairseq.tasks.setup_task(args)

    return task

def optimize_ensemble(models, args):
    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if torch.cuda.is_available() and not args.cpu:
            model.cuda()

    assert len(models) > 0

def load_ensemble(task, parsed_args, optimize=True, override=False):
    if task is None:
        task = setup_task(parsed_args)

    # load ensemble
    models, args = fairseq.checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides={"tokens_per_sample": parsed_args.tokens_per_sample} if override else None,
        task=task
    )
    # NOTE: normally reads parameters like tokens_per_sample from model. Overridden for testing purposes.

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = fairseq.tasks.setup_task(args)

    if optimize:
        optimize_ensemble(models, args)

    return models, args

def load_dataset(task, args):
    # Load dataset splits
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    if args.context_window > 0:
        dataset = fairseq.data.LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad()
        )

    return dataset

def batch_iterator(task, models, args, dataset, shuffle=False):
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=fairseq.utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=shuffle)

    return itr