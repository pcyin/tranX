# coding=utf-8
from __future__ import print_function

import json
import os
import pickle
import sys
from itertools import chain

import numpy as np

from asdl.asdl import ASDLGrammar
from asdl.hypothesis import Hypothesis
from datasets.wikisql.lib.common import detokenize
from datasets.wikisql.lib.dbengine import DBEngine
from datasets.wikisql.lib.query import Query
from asdl.lang.sql.sql_transition_system import SqlTransitionSystem, sql_query_to_asdl_ast, asdl_ast_to_sql_query
from datasets.wikisql.utils import my_detokenize, find_sub_sequence
from asdl.transition_system import GenTokenAction
from components.action_info import ActionInfo
from components.vocab import VocabEntry, Vocab
from model.wikisql.dataset import WikiSqlExample, WikiSqlTable, TableColumn


def get_action_infos(src_query, tgt_actions, force_copy=False, copy_method='token'):
    action_infos = []
    hyp = Hypothesis()
    t = 0
    while t < len(tgt_actions):
        action = tgt_actions[t]

        if type(action) is GenTokenAction:
            begin_t = t
            t += 1
            while t < len(tgt_actions) and type(tgt_actions[t]) is GenTokenAction:
                t += 1
            end_t = t

            gen_token_actions = tgt_actions[begin_t: end_t]
            assert gen_token_actions[-1].is_stop_signal()

            tokens = [action.token for action in gen_token_actions[:-1]]

            try:
                tok_src_start_idx, tok_src_end_idx = find_sub_sequence(src_query, tokens)
                tok_src_idxs = list(range(tok_src_start_idx, tok_src_end_idx))
            except IndexError:
                print('\tCannot find [%s] in [%s]' % (' '.join(tokens), ' '.join(src_query)), file=sys.stderr)
                tok_src_idxs = [src_query.index(token) for token in tokens]

            tok_src_idxs.append(-1)  # for </primitive>

            for tok_src_idx, gen_token_action in zip(tok_src_idxs, gen_token_actions):
                action_info = ActionInfo(gen_token_action)

                if not gen_token_action.is_stop_signal():
                    action_info.copy_from_src = True
                    action_info.src_token_position = tok_src_idx

                    assert src_query[tok_src_idx] == gen_token_action.token

                if hyp.frontier_node:
                    action_info.parent_t = hyp.frontier_node.created_time
                    action_info.frontier_prod = hyp.frontier_node.production
                    action_info.frontier_field = hyp.frontier_field.field

                hyp.apply_action(gen_token_action)
                action_infos.append(action_info)
        else:
            action_info = ActionInfo(action)
            if hyp.frontier_node:
                action_info.parent_t = hyp.frontier_node.created_time
                action_info.frontier_prod = hyp.frontier_node.production
                action_info.frontier_field = hyp.frontier_field.field

            hyp.apply_action(action)
            action_infos.append(action_info)
            t += 1

    # for t, action in enumerate(tgt_actions):
    #     action_info = ActionInfo(action)
    #     action_info.t = t
    #     if hyp.frontier_node:
    #         action_info.parent_t = hyp.frontier_node.created_time
    #         action_info.frontier_prod = hyp.frontier_node.production
    #         action_info.frontier_field = hyp.frontier_field.field
    #
    #     if type(action) is GenTokenAction:
    #         try:
    #             tok_src_idx = src_query.index(str(action.token))
    #             action_info.copy_from_src = True
    #             action_info.src_token_position = tok_src_idx
    #         except ValueError:
    #             if force_copy and not action.is_stop_signal():
    #                 raise ValueError('cannot copy primitive token %s from source' % action.token)
    #
    #     hyp.apply_action(action)
    #     action_infos.append(action_info)

    return action_infos


def load_dataset(transition_system, dataset_file, table_file):
    examples = []
    engine = DBEngine(dataset_file[:-len('jsonl')] + 'db')

    # load table
    tables = dict()
    for line in open(table_file):
        table_entry = json.loads(line)
        tables[table_entry['id']] = table_entry

    for idx, line in enumerate(open(dataset_file)):
        # if idx > 100: break
        entry = json.loads(line)
        del entry['seq_input']
        del entry['seq_output']
        del entry['where_output']

        query = Query.from_tokenized_dict(entry['query'])
        query = query.lower()

        tokenized_conditions = []
        for col, op, val_entry in entry['query']['conds']:
            val = []
            for word, after in zip(val_entry['words'], val_entry['after']):
                val.append(word)

            tokenized_conditions.append([col, op, ' '.join(val)])
        tokenized_query = Query(sel_index=entry['query']['sel'], agg_index=entry['query']['agg'], conditions=tokenized_conditions)

        asdl_ast = sql_query_to_asdl_ast(tokenized_query, transition_system.grammar)
        asdl_ast.sanity_check()
        actions = transition_system.get_actions(asdl_ast)
        hyp = Hypothesis()

        question_tokens = entry['question']['words']
        tgt_action_infos = get_action_infos(question_tokens, actions, force_copy=True)

        for action, action_info in zip(actions, tgt_action_infos):
            assert action == action_info.action
            hyp.apply_action(action)

        reconstructed_query_from_hyp = asdl_ast_to_sql_query(hyp.tree)
        reconstructed_query = asdl_ast_to_sql_query(asdl_ast)

        assert tokenized_query == reconstructed_query

        # now we make sure the tokenized query executes to the same results as the original one!

        detokenized_conds_from_reconstr_query = []
        error = False
        for i, (col, op, val) in enumerate(reconstructed_query_from_hyp.conditions):
            val_tokens = val.split(' ')
            cond_entry = entry['query']['conds'][i]

            assert col == cond_entry[0]
            assert op == cond_entry[1]

            detokenized_cond_val = my_detokenize(val_tokens, entry['question'])
            raw_cond_val = detokenize(cond_entry[2])
            if detokenized_cond_val.lower() != raw_cond_val.lower():
                # print(idx + 1, detokenized_cond_val, raw_cond_val, file=sys.stderr)
                error = True

            detokenized_conds_from_reconstr_query.append((col, op, detokenized_cond_val))

        detokenized_reconstr_query_from_hyp = Query(sel_index=reconstructed_query_from_hyp.sel_index,
                                                    agg_index=reconstructed_query_from_hyp.agg_index,
                                                    conditions=detokenized_conds_from_reconstr_query)

        # make sure the execution result is the same
        hyp_query_result = engine.execute_query(entry['table_id'], detokenized_reconstr_query_from_hyp)
        ref_result = engine.execute_query(entry['table_id'], query)

        if hyp_query_result != ref_result:
            print('[%d]: %s, %s' % (idx, query, detokenized_reconstr_query_from_hyp), file=sys.stderr)

        header = [TableColumn(name=detokenize(col_name), tokens=col_name['words'], type=col_type) for (col_name, col_type) in
                  zip(entry['table']['header'], tables[entry['table_id']]['types'])]
        table = WikiSqlTable(header=header)

        example = WikiSqlExample(idx=idx,
                                 question=question_tokens,
                                 table=table,
                                 tgt_actions=tgt_action_infos,
                                 tgt_code=query,
                                 tgt_ast=asdl_ast,
                                 meta=entry)

        examples.append(example)

        # print(query)

    return examples


def prepare_dataset(data_path):
    grammar = ASDLGrammar.from_text(open('asdl/lang/sql/sql_asdl.txt').read())
    transition_system = SqlTransitionSystem(grammar)

    datasets = []
    for file in ['dev', 'test', 'train']:
        print('processing %s' % file, file=sys.stderr)
        dataset_path = os.path.join(data_path, file + '.jsonl')
        table_path = os.path.join(data_path, file + '.tables.jsonl')
        dataset = load_dataset(transition_system, dataset_path, table_path)
        pickle.dump(dataset, open('data/wikisql/%s.bin' % file, 'wb'))

        datasets.append(dataset)

    train_set = datasets[2]
    dev_set = datasets[0]
    test_set = datasets[1]
    # generate vocabulary
    src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_set], size=100000, freq_cutoff=2)
    primitive_vocab = VocabEntry()
    primitive_vocab.add('</primitive>')

    vocab = Vocab(source=src_vocab, primitive=primitive_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    pickle.dump(vocab, open('data/wikisql/vocab.bin', 'wb'))

    action_len = [len(e.tgt_actions) for e in chain(train_set, dev_set, test_set)]
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)


if __name__ == '__main__':
    # play ground..
    # data_file = '/Users/yinpengcheng/Research/SemanticParsing/WikiSQL/data/dev.jsonl'
    # engine = DBEngine('/Users/yinpengcheng/Research/SemanticParsing/WikiSQL/data/dev.db')
    # for line in open(data_file):
    #     example = json.loads(line)
    #     query = Query.from_dict(example['sql'])
    #     result = engine.execute_query(example['table_id'], query)
    #     pass

    prepare_dataset(data_path='/Users/yinpengcheng/Research/SemanticParsing/WikiSQL/annotated')
