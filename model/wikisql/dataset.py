# coding=utf-8

from itertools import chain
from collections import namedtuple

from components.dataset import Example, Batch

import torch
from torch.autograd import Variable


TableColumn = namedtuple('TableColumn', ['name', 'tokens', 'type'])


class WikiSqlExample(Example):
    def __init__(self, question, table, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        super(WikiSqlExample, self).__init__(question, tgt_actions, tgt_code, tgt_ast, idx, meta)
        self.table = table


class WikiSqlTable(object):
    def __init__(self, header):
        self.header = header


class WikiSqlBatch(Batch):
    def __init__(self, examples, grammar, vocab, cuda=False):
        super(WikiSqlBatch, self).__init__(examples, grammar, vocab, cuda=cuda, copy=True)

    def init_index_tensors(self):
        pass

    def table_head_input_tensor(self):
        if not hasattr(self, '_table_head_input_tensor'):
            _table_head_input_tensor = WikiSqlBatch.get_table_header_input_tensor([e.table for e in self.examples],
                                                                                   self.vocab.source,
                                                                                   cuda=self.cuda)

            setattr(self, '_table_head_input_tensor', _table_head_input_tensor)
        else:
            return self._table_head_input_tensor

    @staticmethod
    def get_table_header_mask(tables, attention_mask=True, cuda=False):
        """
        return: (batch_size, max_column_num)
        """
        T = torch.cuda if cuda else torch
        max_colum_num = max(len(table.header) for table in tables)
        mask_val = 1 if attention_mask else 0
        table_header_mask = [[1 - mask_val] * len(table.header) + [mask_val] * (max_colum_num - len(table.header)) for table in tables]

        return T.ByteTensor(table_header_mask)

    @staticmethod
    def get_table_header_input_tensor(tables, vocab, pad_col_lens=True, cuda=False):
        """
        # table_head_wids: (batch_size, max_column_num, max_head_word_num)
        # table_head_mask: (batch_size, max_column_num)
        # table_col_lens: (batch_size, max_column_num)
        """
        max_column_num = max(len(table.header) for table in tables)
        max_column_word_num = max(len(column.tokens) + 2 for column in chain.from_iterable(table.header for table in tables))
        table_header_wids = []
        table_col_lens = [[len(column.tokens) + 2 for column in table.header] for table in tables]

        if pad_col_lens:
            for lens in table_col_lens:
                lens.extend([0] * (max_column_num - len(lens)))

        for table in tables:
            cur_table_header_wids = []
            for column in table.header:
                cur_header_wids = [vocab['<s>']] + \
                 [vocab[token] for token in column.tokens] + \
                 [vocab['</s>']] + \
                 [vocab['<pad>']] * (max_column_word_num - len(column.tokens) - 2)

                cur_table_header_wids.append(cur_header_wids)

            # pad column
            for i in range(max_column_num - len(table.header)):
                cur_table_header_wids.append([vocab['<pad>']] * max_column_word_num)

            table_header_wids.append(cur_table_header_wids)

        T = torch.cuda if cuda else torch
        table_header_wids = Variable(T.LongTensor(table_header_wids))

        return table_header_wids, table_col_lens
