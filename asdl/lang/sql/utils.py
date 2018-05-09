# coding=utf-8
import re
from babel.numbers import parse_decimal, NumberFormatError

from .lib.query import Query


num_re = re.compile(r'[-+]?\d*\.\d+|\d+')


def my_detokenize(tokens, token_dict, raise_error=False):
    literal = []

    try:
        start_idx, end_idx = find_sub_sequence(token_dict['words'], tokens)
        for idx in range(start_idx, end_idx):
            literal.extend([token_dict['gloss'][idx], token_dict['after'][idx]])

        val = ''.join(literal).strip()
    except IndexError:
        if raise_error:
            raise IndexError('cannot find the entry for [%s] in the token dict [%s]' % (' '.join(tokens),
                                                                                        ' '.join(token_dict['words'])))

        # if we cannot find a consecutive sequence in the gloss dictionary,
        # revert to token-wise matching

        for token in tokens:
            match = False
            for word, gloss, after in zip(token_dict['words'], token_dict['gloss'], token_dict['after']):
                if token == word:
                    literal.extend([gloss, after])
                    match = True
                    break

            if not match and raise_error:
                raise IndexError('cannot find the entry for [%s] in the token dict [%s]' % (' '.join(tokens),
                                                                                            ' '.join(
                                                                                                token_dict['words'])))
        val = ''.join(literal).strip()

    return val


def detokenize_query(query, example_dict, table):
    detokenized_conds = []
    for i, (col, op, val) in enumerate(query.conditions):
        val_tokens = val.split(' ')

        detokenized_cond_val = my_detokenize(val_tokens, example_dict['question'])

        if table.header[col].type == 'real' and not isinstance(detokenized_cond_val, (int, float)):
            if ',' not in detokenized_cond_val:
                try:
                    detokenized_cond_val = float(parse_decimal(val))
                except NumberFormatError as e:
                    try: detokenized_cond_val = float(num_re.findall(val)[0])
                    except: pass

        detokenized_conds.append((col, op, detokenized_cond_val))

    detokenized_query = Query(sel_index=query.sel_index, agg_index=query.agg_index, conditions=detokenized_conds)

    return detokenized_query


def find_sub_sequence(sequence, query_seq):
    for i in range(len(sequence)):
        if sequence[i: len(query_seq) + i] == query_seq:
            return i, len(query_seq) + i

    raise IndexError