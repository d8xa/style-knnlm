def build_command(command_strings):
    import fairseq, sys
    sys.argv = command_strings
    parser = fairseq.options.get_eval_lm_parser()
    parsed_args = fairseq.options.parse_args_and_arch(parser)
    
    return parsed_args