# coding=utf-8
from __future__ import print_function

import ast
import sys
import traceback

import astor

from asdl.lang.py.py_asdl_helper import asdl_ast_to_python_ast, python_ast_to_asdl_ast
from asdl.lang.py.py_utils import tokenize_code as tokenize_py_code


def decode(examples, model, args, verbose=False):
    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    count = 0
    for example in examples:
        hyps = model.parse(example.src_sent, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps[:args.sample_size]):
            try:
                py_ast = asdl_ast_to_python_ast(hyp.tree, model.grammar)
                code = astor.to_source(py_ast).strip()
                decoded_hyps.append((hyp, code))
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
    for example, hyps in zip(examples, decode_results):
        if hyps:
            ref_code = example.tgt_code
            ref_py_ast = ast.parse(ref_code).body[0]
            ref_reformatted_code = astor.to_source(ref_py_ast).strip()

            cur_oracle = 0.
            hyp_code_set = set()
            for hyp_id, (hyp, hyp_code) in enumerate(hyps):
                try:
                    ref_code_tokens = tokenize_py_code(ref_reformatted_code)
                    hyp_code_tokens = tokenize_py_code(hyp_code)
                except:
                    print('Hyp Id [%d] error in tokenizing [%s]' % (hyp_id, hyp_code), file=sys.stderr)
                    continue

                if hyp_code in hyp_code_set:
                    print('Duplicate Hyp Example [%d], Code %s' % (example.idx, hyp_code), file=sys.stdout)
                hyp_code_set.add(hyp_code)

                if hyp_id == 0 and hyp_code_tokens == ref_code_tokens:
                    cum_acc += 1

                if cur_oracle == 0. and hyp_code_tokens == ref_code_tokens:
                    cur_oracle = 1.

            cum_oracle_acc += cur_oracle

    eval_result = {'accuracy': cum_acc / len(examples),
                   'oracle_accuracy': cum_oracle_acc / len(examples)}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
