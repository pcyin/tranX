# coding=utf-8

from __future__ import print_function

import re
import cPickle as pickle
import ast
import astor
import nltk
import sys

import numpy as np

from asdl.asdl_ast import RealizedField
from asdl.lang.py.py_asdl_helper import python_ast_to_asdl_ast, asdl_ast_to_python_ast
from asdl.lang.py.py_transition_system import PythonTransitionSystem
from asdl.hypothesis import *

from components.action_info import ActionInfo

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")


class Django(object):
    @staticmethod
    def canonicalize_code(code):
        if p_elif.match(code):
            code = 'if True: pass\n' + code

        if p_else.match(code):
            code = 'if True: pass\n' + code

        if p_try.match(code):
            code = code + 'pass\nexcept: pass'
        elif p_except.match(code):
            code = 'try: pass\n' + code
        elif p_finally.match(code):
            code = 'try: pass\n' + code

        if p_decorator.match(code):
            code = code + '\ndef dummy(): pass'

        if code[-1] == ':':
            code = code + 'pass'

        return code

    @staticmethod
    def canonicalize_query(query):
        """
        canonicalize the query, replace strings to a special place holder
        """
        str_count = 0
        str_map = dict()

        matches = QUOTED_STRING_RE.findall(query)
        # de-duplicate
        cur_replaced_strs = set()
        for match in matches:
            # If one or more groups are present in the pattern,
            # it returns a list of groups
            quote = match[0]
            str_literal = quote + match[1] + quote

            if str_literal in cur_replaced_strs:
                continue

            # FIXME: substitute the ' % s ' with
            if str_literal in ['\'%s\'', '\"%s\"']:
                continue

            str_repr = '_STR:%d_' % str_count
            str_map[str_literal] = str_repr

            query = query.replace(str_literal, str_repr)

            str_count += 1
            cur_replaced_strs.add(str_literal)

        # tokenize
        query_tokens = nltk.word_tokenize(query)

        new_query_tokens = []
        # break up function calls like foo.bar.func
        for token in query_tokens:
            new_query_tokens.append(token)
            i = token.find('.')
            if 0 < i < len(token) - 1:
                new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
                new_query_tokens.extend(new_tokens)

        query = ' '.join(new_query_tokens)

        return query, str_map

    @staticmethod
    def canonicalize_example(query, code):

        canonical_query, str_map = Django.canonicalize_query(query)
        query_tokens = canonical_query.split(' ')
        canonical_code = code

        for str_literal, str_repr in str_map.iteritems():
            canonical_code = canonical_code.replace(str_literal, '\'' + str_repr + '\'')

        canonical_code = Django.canonicalize_code(canonical_code)

        # sanity check
        try:
            gold_ast_tree = ast.parse(canonical_code).body[0]
        except:
            print('error!')
            canonical_code = Django.canonicalize_code(code)
            gold_ast_tree = ast.parse(canonical_code).body[0]
            str_map = {}

        # parse_tree = python_ast_to_asdl_ast(gold_ast_tree, grammar)
        # gold_source = astor.to_source(gold_ast_tree)
        # ast_tree = asdl_ast_to_python_ast(parse_tree, grammar)
        # source = astor.to_source(ast_tree)

        # assert gold_source == source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, source)
        #
        # # action check
        # parser = PythonTransitionSystem(grammar)
        # actions = parser.get_actions(parse_tree)
        #
        # hyp = Hypothesis()
        # for action in actions:
        #     assert action.__class__ in parser.get_valid_continuation_types(hyp)
        #     if isinstance(action, ApplyRuleAction):
        #         assert action in parser.get_valid_continuations(hyp)
        #     hyp.apply_action(action)
        #
        # src_from_hyp = astor.to_source(asdl_ast_to_python_ast(hyp.tree, grammar))
        # assert src_from_hyp == gold_source

        return query_tokens, canonical_code, str_map

    @staticmethod
    def parse_django_dataset(annot_file, code_file, asdl_file_path, MAX_QUERY_LENGTH=70):
        asdl_text = open(asdl_file_path).read()
        grammar = ASDLGrammar.from_text(asdl_text)
        transition_system = PythonTransitionSystem(grammar)

        loaded_examples = []

        from components.vocab import Vocab, VocabEntry
        from components.dataset import Example

        for idx, (src_query, tgt_code) in enumerate(zip(open(annot_file), open(code_file))):
            src_query = src_query.strip()
            tgt_code = tgt_code.strip()

            src_query_tokens, tgt_canonical_code, str_map = Django.canonicalize_example(src_query, tgt_code)
            python_ast = ast.parse(tgt_canonical_code).body[0]
            gold_source = astor.to_source(python_ast)
            tgt_ast = python_ast_to_asdl_ast(python_ast, grammar)
            tgt_actions = transition_system.get_actions(tgt_ast)

            # sanity check
            hyp = Hypothesis()
            for action in tgt_actions:
                assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
                if isinstance(action, ApplyRuleAction):
                    assert action.production in transition_system.get_valid_continuating_productions(hyp)
                hyp = hyp.clone_and_apply_action(action)

            src_from_hyp = astor.to_source(asdl_ast_to_python_ast(hyp.tree, grammar))
            assert src_from_hyp == gold_source

            loaded_examples.append({'src_query_tokens': src_query_tokens,
                                    'tgt_canonical_code': tgt_canonical_code,
                                    'tgt_ast': tgt_ast,
                                    'tgt_actions': tgt_actions,
                                    'raw_code': tgt_code, 'str_map': str_map})

            print('first pass, processed %d' % idx, file=sys.stderr)

        src_vocab = VocabEntry.from_corpus([e['src_query_tokens'] for e in loaded_examples], size=5000, freq_cutoff=3)

        primitive_tokens = [map(lambda a: a.token,
                               filter(lambda a: isinstance(a, GenTokenAction), e['tgt_actions']))
                            for e in loaded_examples]

        primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=3)
        assert '_STR:0_' in primitive_vocab

        vocab = Vocab(source=src_vocab, primitive=primitive_vocab)
        print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

        train_examples = []
        dev_examples = []
        test_examples = []

        action_len = []

        for idx, e in enumerate(loaded_examples):
            src_query_tokens = e['src_query_tokens'][:MAX_QUERY_LENGTH]
            tgt_actions = e['tgt_actions']
            tgt_action_infos = Django.get_action_infos(src_query_tokens, tgt_actions)

            example = Example(idx=idx,
                              src_sent=src_query_tokens,
                              tgt_actions=tgt_action_infos,
                              tgt_code=e['tgt_canonical_code'],
                              tgt_ast=e['tgt_ast'],
                              meta={'raw_code': e['raw_code'], 'str_map': e['str_map']})

            print('second pass, processed %d' % idx, file=sys.stderr)

            action_len.append(len(tgt_action_infos))

            # train, valid, test split
            if 0 <= idx < 16000:
                train_examples.append(example)
            elif 16000 <= idx < 17000:
                dev_examples.append(example)
            else:
                test_examples.append(example)

        print('Max action len: %d' % max(action_len), file=sys.stderr)
        print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
        print('Actions larger than 100: %d' % len(filter(lambda x: x > 100, action_len)), file=sys.stderr)

        return (train_examples, dev_examples, test_examples), vocab

    @staticmethod
    def get_action_infos(src_query, tgt_actions):
        action_infos = []
        hyp = Hypothesis()
        for t, action in enumerate(tgt_actions):
            action_info = ActionInfo(action)
            action_info.t = t
            if hyp.frontier_node:
                action_info.parent_t = hyp.frontier_node.created_time
                action_info.frontier_prod = hyp.frontier_node.production
                action_info.frontier_field = hyp.frontier_field.field

            if isinstance(action, GenTokenAction):
                try:
                    tok_src_idx = src_query.index(str(action.token))
                    action_info.copy_from_src = True
                    action_info.src_token_position = tok_src_idx
                except ValueError:
                    pass

            hyp.apply_action(action)
            action_infos.append(action_info)

        return action_infos

    @staticmethod
    def generate_django_dataset():
        annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
        code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'

        (train, dev, test), vocab = Django.parse_django_dataset(annot_file, code_file, 'asdl/lang/py/py_asdl.txt')

        pickle.dump(train, open('data/django/train.bin', 'w'))
        pickle.dump(dev, open('data/django/dev.bin', 'w'))
        pickle.dump(test, open('data/django/test.bin', 'w'))
        pickle.dump(vocab, open('data/django/vocab.bin', 'w'))

    @staticmethod
    def run():
        asdl_text = open('asdl/lang/py/py_asdl.txt').read()
        grammar = ASDLGrammar.from_text(asdl_text)

        annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
        code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'

        transition_system = PythonTransitionSystem(grammar)

        for idx, (src_query, tgt_code) in enumerate(zip(open(annot_file), open(code_file))):
            src_query = src_query.strip()
            tgt_code = tgt_code.strip()

            query_tokens, tgt_canonical_code, str_map = Django.canonicalize_example(src_query, tgt_code)
            python_ast = ast.parse(tgt_canonical_code).body[0]
            gold_source = astor.to_source(python_ast)
            tgt_ast = python_ast_to_asdl_ast(python_ast, grammar)
            tgt_actions = transition_system.get_actions(tgt_ast)

            # sanity check
            hyp = Hypothesis()
            for action in tgt_actions:
                assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
                if isinstance(action, ApplyRuleAction):
                    assert action.production in transition_system.get_valid_continuating_productions(hyp)
                hyp = hyp.clone_and_apply_action(action)

            src_from_hyp = astor.to_source(asdl_ast_to_python_ast(hyp.tree, grammar))
            assert src_from_hyp == gold_source

            print(idx)


if __name__ == '__main__':
    # Django.run()
    # f1 = Field('hahah', ASDLPrimitiveType('123'), 'single')
    # rf1 = RealizedField(f1, value=123)
    #
    # # print(f1 == rf1)
    # a = {f1: 1}
    # print(a[rf1])
    Django.generate_django_dataset()
