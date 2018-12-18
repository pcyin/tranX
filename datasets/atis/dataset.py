# coding=utf-8
from __future__ import print_function

import sys
import numpy as np
try: import cPickle as pickle
except: import pickle

from asdl.hypothesis import Hypothesis
from components.action_info import get_action_infos
from asdl.transition_system import *
from components.dataset import Example
from components.vocab import VocabEntry, Vocab
from asdl.lang.lambda_dcs.lambda_dcs_transition_system import *
from asdl.lang.lambda_dcs.logical_form import *


def load_dataset(transition_system, dataset_file, reorder_predicates=True):
    examples = []
    for idx, line in enumerate(open(dataset_file)):
        src_query, tgt_code = line.strip().split('\t')

        src_query_tokens = src_query.split(' ')

        lf = parse_lambda_expr(tgt_code)
        assert lf.to_string() == tgt_code

        if reorder_predicates:
            ordered_lf = get_canonical_order_of_logical_form(lf, order_by='alphabet')
            assert ordered_lf == lf
            lf = ordered_lf

        gold_source = lf.to_string()

        tgt_ast = logical_form_to_ast(grammar, lf)
        reconstructed_lf = ast_to_logical_form(tgt_ast)
        assert lf == reconstructed_lf

        tgt_actions = transition_system.get_actions(tgt_ast)

        print(idx)
        print('Utterance: %s' % src_query)
        print('Reference: %s' % tgt_code)
        # print('===== Actions =====')
        # sanity check
        hyp = Hypothesis()
        for action in tgt_actions:
            assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
            if isinstance(action, ApplyRuleAction):
                assert action.production in transition_system.get_valid_continuating_productions(hyp)
            hyp = hyp.clone_and_apply_action(action)
            # print(action)

        assert hyp.frontier_node is None and hyp.frontier_field is None

        src_from_hyp = transition_system.ast_to_surface_code(hyp.tree)
        assert src_from_hyp == gold_source

        tgt_action_infos = get_action_infos(src_query_tokens, tgt_actions)

        # print(' '.join(src_query_tokens))
        print('***')
        print(lf.to_string())
        print()
        example = Example(idx=idx,
                          src_sent=src_query_tokens,
                          tgt_actions=tgt_action_infos,
                          tgt_code=gold_source,
                          tgt_ast=tgt_ast,
                          meta=None)

        examples.append(example)

    return examples


def prepare_atis_dataset():
    vocab_freq_cutoff = 2
    grammar = ASDLGrammar.from_text(open('asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    transition_system = LambdaCalculusTransitionSystem(grammar)

    train_set = load_dataset(transition_system, 'data/atis/train.txt')
    dev_set = load_dataset(transition_system, 'data/atis/dev.txt')
    test_set = load_dataset(transition_system, 'data/atis/test.txt')

    # generate vocabulary
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)

    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_set]

    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=0)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_set]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=2)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, dev_set, test_set)]
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(filter(lambda x: x > 100, action_len)), file=sys.stderr)

    pickle.dump(train_set, open('data/atis/train.bin', 'w'))
    pickle.dump(dev_set, open('data/atis/dev.bin', 'w'))
    pickle.dump(test_set, open('data/atis/test.bin', 'w'))
    pickle.dump(vocab, open('data/atis/vocab.freq%d.bin' % vocab_freq_cutoff, 'w'))


def prepare_geo_dataset():
    vocab_freq_cutoff = 2 # for geo query
    grammar = ASDLGrammar.from_text(open('asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    transition_system = LambdaCalculusTransitionSystem(grammar)

    train_set = load_dataset(transition_system, 'data/geo/train.txt', reorder_predicates=False)
    test_set = load_dataset(transition_system, 'data/geo/test.txt', reorder_predicates=False)

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

    pickle.dump(train_set, open('data/geo/train.bin', 'wb'))

    # randomly sample 20% data as the dev set
    np.random.seed(1234)
    dev_example_ids = np.random.choice(list(range(len(train_set))), replace=False, size=int(0.2 * len(train_set)))
    train_example_ids = [i for i in range(len(train_set)) if i not in dev_example_ids]

    print('# splitted train examples %d, # splitted dev examples %d' % (len(train_example_ids), len(dev_example_ids)), file=sys.stderr)

    dev_set = [train_set[i] for i in dev_example_ids]
    train_set = [train_set[i] for i in train_example_ids]

    pickle.dump(train_set, open('data/geo/train.split.bin', 'wb'))
    pickle.dump(dev_set, open('data/geo/dev.bin', 'wb'))

    pickle.dump(test_set, open('data/geo/test.bin', 'wb'))
    pickle.dump(vocab, open('data/geo/vocab.freq2.bin', 'wb'))


def generate_vocab_for_paraphrase_model(vocab_path, save_path):
    from components.vocab import VocabEntry, Vocab

    vocab = pickle.load(open(vocab_path))
    para_vocab = VocabEntry()
    for i in range(0, 10):
        para_vocab.add('<unk_%d>' % i)
    for word in vocab.source.word2id:
        para_vocab.add(word)
    for word in vocab.code.word2id:
        para_vocab.add(word)

    pickle.dump(para_vocab, open(save_path, 'w'))


if __name__ == '__main__':
    grammar = ASDLGrammar.from_text(open('asdl/lang/lambda_dcs/lambda_asdl.txt').read())
    transition_system = LambdaCalculusTransitionSystem(grammar)
    # load_dataset(transition_system, 'data/atis/train.txt')
    # prepare_geo_dataset()
    # prepare_atis_dataset()
    generate_vocab_for_paraphrase_model('data/atis/vocab.freq2.bin', 'data/atis/vocab.para.freq2.bin')
