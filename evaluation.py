# coding=utf-8
from __future__ import print_function

import sys
import traceback
import os


def decode(examples, model, args, verbose=False):
    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    count = 0
    for example in examples:
        if args.lang == 'wikisql':
            hyps = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size)
        else:
            hyps = model.parse(example.src_sent, context=None, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('example id: %d, hypothesis id: %d' % (example.idx, hyp_id), file=sys.stdout)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1
        if verbose and count % 50 == 0:
            print('decoded %d examples...' % count, file=sys.stdout)

        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, args, verbose=False, return_decode_result=False):
    cum_oracle_acc = cum_acc = 0.0
    decode_results = decode(examples, parser, args, verbose=verbose)

    if args.lang == 'wikisql':
        from asdl.lang.sql.lib.dbengine import DBEngine
        from asdl.lang.sql.utils import detokenize_query

        if args.mode == 'train':
            table_file = os.path.splitext(args.dev_file)[0] + '.db'
        else:
            table_file = os.path.splitext(args.test_file)[0] + '.db'
        execution_engine = DBEngine(table_file)

    for example, hyps in zip(examples, decode_results):
        if hyps:
            cur_oracle = 0.
            hyp_code_set = set()
            # print('Reference: %s' % example.tgt_code, file=sys.stderr)
            for hyp_id, hyp in enumerate(hyps):
                try:
                    if args.lang == 'wikisql':
                        result = parser.transition_system.hyp_correct(hyp, example, execution_engine)
                        # print('Hyp %d: %s ||| %s' % (hyp_id, detokenize_query(hyp.code, example.meta, example.table), result),
                        #       file=sys.stderr)
                    else:
                        result = parser.transition_system.hyp_correct(hyp, example)

                    if hyp_id == 0 and result:
                        cum_acc += 1
                    if cur_oracle == 0. and result:
                        cur_oracle = 1.

                    hyp.correct = result
                except:
                    print('Hyp Id [%d] error in evluating [%s]' % (hyp_id, hyp.code), file=sys.stderr)
                    hyp.correct = False
                    continue

                if args.lang in ['lambda_dcs', 'python']:
                    if hyp.code in hyp_code_set:
                        print('Duplicate Hyp Example [%d], Code %s' % (example.idx, hyp.code), file=sys.stdout)
                    hyp_code_set.add(hyp.code)

            cum_oracle_acc += cur_oracle

    eval_result = {'accuracy': cum_acc / len(examples),
                   'oracle_accuracy': cum_oracle_acc / len(examples)}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
