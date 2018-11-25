from __future__ import print_function

import sys, traceback
import numpy as np
from common.registerable import Registrable


@Registrable.register('default_evaluator')
class Evaluator(object):
    def __init__(self, transition_system=None):
        self.transition_system = transition_system
        self.default_metric = 'accuracy'

    def is_hyp_correct(self, example, hyp):
        return self.transition_system.compare_ast(hyp.tree, example.tgt_ast)

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        correct_array = []
        oracle_array = []
        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):
                    try:
                        is_correct = self.is_hyp_correct(example, hyp)
                    except:
                        is_correct = False

                        print('-' * 60, file=sys.stdout)
                        print('Error in evaluating Example %s, hyp %d {{ %s }}' % (example.idx, hyp_id, hyp.code),
                              file=sys.stdout)

                        print('example id: %s, hypothesis id: %d' % (example.idx, hyp_id), file=sys.stdout)
                        traceback.print_exc(file=sys.stdout)
                        print('-' * 60, file=sys.stdout)

                    hyp.is_correct = is_correct

                correct_array.append(hyp_list[0].is_correct)
                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))
            else:
                correct_array.append(False)
                oracle_array.append(False)

        acc = np.average(correct_array)
        if fast_mode:
            return acc

        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            oracle_accuracy=oracle_acc)

        return eval_results


@Registrable.register('cached_evaluator')
class CachedExactMatchEvaluator(Evaluator):
    def is_hyp_correct(self, example, hyp):
        raise hyp.is_correct

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        if fast_mode:
            acc = sum(hyps[0].is_correct for hyps in decode_results if len(hyps) > 0) / float(len(examples))
            return acc

        acc_array = []
        oracle_array = []
        for hyp_list in decode_results:
            acc_array.append(hyp_list[0].is_correct if hyp_list else False)
            oracle_array.append(any(hyp.is_correct for hyp in hyp_list))

        return dict(accuracy=np.average(acc_array),
                    oracle_array=np.average(oracle_array))
