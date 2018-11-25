# coding=utf-8
from __future__ import print_function

import sys
import traceback
from tqdm import tqdm


def decode(examples, model, args, verbose=False, **kwargs):
    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    if args.lang == 'wikisql':
        from asdl.lang.sql.lib.dbengine import DBEngine
        from asdl.lang.sql.utils import detokenize_query

    decode_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if args.lang == 'wikisql':
            hyps = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size)
        else:
            hyps = model.parse(example.src_sent, context=None, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                got_code = True
                if args.lang == 'wikisql' and args.answer_prune:
                    # try execute the code, if fails, skip this example!
                    # if the execution returns null, also skip this example!
                    detokenized_hyp_query = detokenize_query(hyp.code, example.meta, example.table)
                    hyp_answer = kwargs['execution_engine'].execute_query(example.meta['table_id'],
                                                                          detokenized_hyp_query,
                                                                          lower=True)
                    if len(hyp_answer) == 0: continue

                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                             ' '.join(example.src_sent),
                                                                                             example.tgt_code,
                                                                                             hyp_id,
                                                                                             hyp.tree.to_string()), file=sys.stdout)
                    if got_code:
                        print()
                        print(hyp.code)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1

        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    decode_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only)

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
