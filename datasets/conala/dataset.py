import json
import sys
import numpy as np
import pickle

from components.action_info import get_action_infos
from datasets import ConalaEvaluator
from datasets.conala.util import *
from asdl.lang.py3.py3_transition_system import python_ast_to_asdl_ast, asdl_ast_to_python_ast, Python3TransitionSystem

from asdl.hypothesis import *
from asdl.transition_system import *

from components.vocab import Vocab, VocabEntry
from components.dataset import Example
from components.dataset import Dataset
from components.action_info import ActionInfo


def preprocess_conala_dataset(train_file, test_file, grammar_file, src_freq=2, code_freq=2):
    np.random.seed(1234)

    asdl_text = open(grammar_file).read()
    grammar = ASDLGrammar.from_text(asdl_text)
    transition_system = Python3TransitionSystem(grammar)

    print('process training data...')
    train_examples = preprocess_dataset(train_file, name='train', transition_system=transition_system)

    # held out 200 examples for development
    full_train_examples = train_examples[:]
    np.random.shuffle(train_examples)
    dev_examples = train_examples[:200]
    train_examples = train_examples[200:]

    # full_train_examples = train_examples[:]
    # np.random.shuffle(train_examples)
    # dev_examples = []
    # dev_questions = set()
    # dev_examples_id = []
    # for i, example in enumerate(full_train_examples):
    #     qid = example.meta['example_dict']['question_id']
    #     if qid not in dev_questions and len(dev_examples) < 200:
    #         dev_questions.add(qid)
    #         dev_examples.append(example)
    #         dev_examples_id.append(i)

    # train_examples = [e for i, e in enumerate(full_train_examples) if i not in dev_examples_id]
    print(f'{len(train_examples)} training instances', file=sys.stderr)
    print(f'{len(dev_examples)} dev instances', file=sys.stderr)

    print('process testing data...')
    test_examples = preprocess_dataset(test_file, name='test', transition_system=transition_system)
    print(f'{len(test_examples)} testing instances', file=sys.stderr)

    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=5000,
                                       freq_cutoff=src_freq)
    primitive_tokens = [map(lambda a: a.action.token,
                            filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_examples]
    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=code_freq)

    # generate vocabulary for the code tokens!
    code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_examples]
    code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=code_freq)

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_lens = [len(e.tgt_actions) for e in train_examples]
    print('Max action len: %d' % max(action_lens), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_lens), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_lens))), file=sys.stderr)

    pickle.dump(train_examples, open('data/conala/train.var_str_sep.new_dev.bin', 'wb'))
    pickle.dump(full_train_examples, open('data/conala/train.var_str_sep.full.bin', 'wb'))
    pickle.dump(dev_examples, open('data/conala/dev.var_str_sep.new_dev.bin', 'wb'))
    pickle.dump(test_examples, open('data/conala/test.var_str_sep.new_dev.bin', 'wb'))
    pickle.dump(vocab, open('data/conala/vocab.var_str_sep.new_dev.src_freq%d.code_freq%d.bin' % (src_freq, code_freq), 'wb'))


def preprocess_dataset(file_path, transition_system, name='train'):
    dataset = json.load(open(file_path))
    examples = []
    evaluator = ConalaEvaluator(transition_system)

    f = open(file_path + '.debug', 'w')

    for i, example_json in enumerate(dataset):
        example_dict = preprocess_example(example_json)
        if example_json['question_id'] in (18351951, 9497290, 19641579, 32283692):
            print(example_json['question_id'])
            continue

        python_ast = ast.parse(example_dict['canonical_snippet'])
        canonical_code = astor.to_source(python_ast).strip()
        tgt_ast = python_ast_to_asdl_ast(python_ast, transition_system.grammar)
        tgt_actions = transition_system.get_actions(tgt_ast)

        # sanity check
        hyp = Hypothesis()
        for t, action in enumerate(tgt_actions):
            assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
            if isinstance(action, ApplyRuleAction):
                assert action.production in transition_system.get_valid_continuating_productions(hyp)

            p_t = -1
            f_t = None
            if hyp.frontier_node:
                p_t = hyp.frontier_node.created_time
                f_t = hyp.frontier_field.field.__repr__(plain=True)

            # print('\t[%d] %s, frontier field: %s, parent: %d' % (t, action, f_t, p_t))
            hyp = hyp.clone_and_apply_action(action)

        assert hyp.frontier_node is None and hyp.frontier_field is None
        hyp.code = code_from_hyp = astor.to_source(asdl_ast_to_python_ast(hyp.tree, transition_system.grammar)).strip()
        assert code_from_hyp == canonical_code

        decanonicalized_code_from_hyp = decanonicalize_code(code_from_hyp, example_dict['slot_map'])
        assert compare_ast(ast.parse(example_json['snippet']), ast.parse(decanonicalized_code_from_hyp))
        assert transition_system.compare_ast(transition_system.surface_code_to_ast(decanonicalized_code_from_hyp),
                                             transition_system.surface_code_to_ast(example_json['snippet']))

        tgt_action_infos = get_action_infos(example_dict['intent_tokens'], tgt_actions)

        example = Example(idx=f'{i}-{example_json["question_id"]}',
                          src_sent=example_dict['intent_tokens'],
                          tgt_actions=tgt_action_infos,
                          tgt_code=canonical_code,
                          tgt_ast=tgt_ast,
                          meta=dict(example_dict=example_json,
                                    slot_map=example_dict['slot_map']))
        assert evaluator.is_hyp_correct(example, hyp)

        examples.append(example)

        # log!
        f.write(f'Example: {example.idx}\n')
        f.write(f"Original Utterance: {example.meta['example_dict']['rewritten_intent']}\n")
        f.write(f"Original Snippet: {example.meta['example_dict']['snippet']}\n")
        f.write(f"\n")
        f.write(f"Utterance: {' '.join(example.src_sent)}\n")
        f.write(f"Snippet: {example.tgt_code}\n")
        f.write(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    f.close()

    return examples


def preprocess_example(example_json):
    intent = example_json['intent']
    rewritten_intent = example_json['rewritten_intent']
    snippet = example_json['snippet']
    question_id = example_json['question_id']

    if rewritten_intent is None:
        rewritten_intent = intent

    canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
    canonical_snippet = canonicalize_code(snippet, slot_map)
    intent_tokens = tokenize_intent(canonical_intent)
    decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)

    reconstructed_snippet = astor.to_source(ast.parse(snippet)).strip()
    reconstructed_decanonical_snippet = astor.to_source(ast.parse(decanonical_snippet)).strip()

    assert compare_ast(ast.parse(reconstructed_snippet), ast.parse(reconstructed_decanonical_snippet))

    return {'canonical_intent': canonical_intent,
            'intent_tokens': intent_tokens,
            'slot_map': slot_map,
            'canonical_snippet': canonical_snippet}


def generate_vocab_for_paraphrase_model(vocab_path, save_path):
    from components.vocab import VocabEntry, Vocab

    vocab = pickle.load(open(vocab_path, 'rb'))
    para_vocab = VocabEntry()
    for i in range(0, 10):
        para_vocab.add('<unk_%d>' % i)
    for word in vocab.source.word2id:
        para_vocab.add(word)
    for word in vocab.code.word2id:
        para_vocab.add(word)

    pickle.dump(para_vocab, open(save_path, 'wb'))


if __name__ == '__main__':
    preprocess_conala_dataset(train_file='/Users/yinpengcheng/Research/SemanticParsing/conala_eval/data/conala-train.json',
                              test_file='/Users/yinpengcheng/Research/SemanticParsing/conala_eval/data/conala-test.json',
                              grammar_file='asdl/lang/py3/py3_asdl.simplified.txt', src_freq=3, code_freq=3)

    # generate_vocab_for_paraphrase_model('data/conala/vocab.src_freq3.code_freq3.bin', 'data/conala/vocab.para.src_freq3.code_freq3.bin')
