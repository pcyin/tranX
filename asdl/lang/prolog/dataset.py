# coding=utf-8
from __future__ import print_function
import sys

from asdl.hypothesis import Hypothesis, ApplyRuleAction
from asdl.lang.prolog.prolog_transition_system import *
from asdl.asdl import ASDLGrammar
from asdl.lang.py.dataset import get_action_infos
from components.dataset import Example
from components.vocab import VocabEntry, Vocab

try: import cPickle as pickle
except: import pickle

import numpy as np


def load_dataset(transition_system, dataset_file):
    examples = []
    for idx, line in enumerate(open(dataset_file)):
        src_query, tgt_code = line.strip().split('\t')

        src_query_tokens = src_query.split(' ')

        tgt_ast = prolog_expr_to_ast(transition_system.grammar, tgt_code)
        reconstructed_prolog_expr = ast_to_prolog_expr(tgt_ast)
        assert tgt_code == reconstructed_prolog_expr

        tgt_actions = transition_system.get_actions(tgt_ast)

        # sanity check
        hyp = Hypothesis()
        for action in tgt_actions:
            assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
            if isinstance(action, ApplyRuleAction):
                assert action.production in transition_system.get_valid_continuating_productions(hyp)
            hyp = hyp.clone_and_apply_action(action)

        assert hyp.frontier_node is None and hyp.frontier_field is None

        assert is_equal_ast(hyp.tree, tgt_ast)

        expr_from_hyp = transition_system.ast_to_surface_code(hyp.tree)
        assert expr_from_hyp == tgt_code

        tgt_action_infos = get_action_infos(src_query_tokens, tgt_actions)

        print(idx)
        example = Example(idx=idx,
                          src_sent=src_query_tokens,
                          tgt_actions=tgt_action_infos,
                          tgt_code=tgt_code,
                          tgt_ast=tgt_ast,
                          meta=None)

        examples.append(example)

    return examples


def prepare_dataset():
    # vocab_freq_cutoff = 1 for atis
    vocab_freq_cutoff = 2  # for geo query
    grammar = ASDLGrammar.from_text(open('asdl/lang/prolog/prolog_asdl.txt').read())
    transition_system = PrologTransitionSystem(grammar)

    train_set = load_dataset(transition_system, 'data/jobs/train.txt')
    test_set = load_dataset(transition_system, 'data/jobs/test.txt')

    # generate vocabulary
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)

    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_set]

    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=0)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_set]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, test_set)]
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open('data/jobs/train.bin', 'wb'))
    pickle.dump(test_set, open('data/jobs/test.bin', 'wb'))
    pickle.dump(vocab, open('data/jobs/vocab.freq2.bin', 'wb'))


if __name__ == '__main__':
    prepare_dataset()
