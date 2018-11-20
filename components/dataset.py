# coding=utf-8
from collections import OrderedDict

import torch
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

from torch.autograd import Variable

from asdl.transition_system import ApplyRuleAction, ReduceAction
from common.utils import cached_property

from model import nn_utils


class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            batch_examples.sort(key=lambda e: -len(e.src_sent))

            yield batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        self.src_sent = src_sent
        self.tgt_code = tgt_code
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta


class Batch(object):
    def __init__(self, examples, grammar, vocab, copy=True, cuda=False):
        self.examples = examples
        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]

        self.grammar = grammar
        self.vocab = vocab
        self.copy = copy
        self.cuda = cuda

        self.init_index_tensors()

    def __len__(self):
        return len(self.examples)

    def get_frontier_field_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.field2id[e.tgt_actions[t].frontier_field])
                # assert self.grammar.id2field[ids[-1]] == e.tgt_actions[t].frontier_field
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def get_frontier_prod_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.prod2id[e.tgt_actions[t].frontier_prod])
                # assert self.grammar.id2prod[ids[-1]] == e.tgt_actions[t].frontier_prod
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def get_frontier_field_type_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.type2id[e.tgt_actions[t].frontier_field.type])
                # assert self.grammar.id2type[ids[-1]] == e.tgt_actions[t].frontier_field.type
            else:
                ids.append(0)

        return Variable(torch.cuda.LongTensor(ids)) if self.cuda else Variable(torch.LongTensor(ids))

    def init_index_tensors(self):
        self.apply_rule_idx_matrix = []
        self.apply_rule_mask = []
        self.primitive_idx_matrix = []
        self.gen_token_mask = []
        self.primitive_copy_mask = []
        self.primitive_copy_token_idx_mask = np.zeros((self.max_action_num, len(self), max(self.src_sents_len)), dtype='float32')

        for t in range(self.max_action_num):
            app_rule_idx_row = []
            app_rule_mask_row = []
            token_row = []
            gen_token_mask_row = []
            copy_mask_row = []

            for e_id, e in enumerate(self.examples):
                app_rule_idx = app_rule_mask = token_idx = gen_token_mask = copy_mask = 0
                if t < len(e.tgt_actions):
                    action = e.tgt_actions[t].action
                    action_info = e.tgt_actions[t]

                    if isinstance(action, ApplyRuleAction):
                        app_rule_idx = self.grammar.prod2id[action.production]
                        # assert self.grammar.id2prod[app_rule_idx] == action.production
                        app_rule_mask = 1
                    elif isinstance(action, ReduceAction):
                        app_rule_idx = len(self.grammar)
                        app_rule_mask = 1
                    else:
                        src_sent = self.src_sents[e_id]
                        token = str(action.token)
                        token_idx = self.vocab.primitive[action.token]

                        token_can_copy = False

                        if self.copy and token in src_sent:
                            token_pos_list = [idx for idx, _token in enumerate(src_sent) if _token == token]
                            self.primitive_copy_token_idx_mask[t, e_id, token_pos_list] = 1.
                            copy_mask = 1
                            token_can_copy = True

                        if token_can_copy is False or token_idx != self.vocab.primitive.unk_id:
                            # if the token is not copied, we can only generate this token from the vocabulary,
                            # even if it is a <unk>.
                            # otherwise, we can still generate it from the vocabulary
                            gen_token_mask = 1

                        if token_can_copy:
                            assert action_info.copy_from_src
                            assert action_info.src_token_position in token_pos_list

                        # # cannot copy, only generation
                        # # could be unk!
                        # if not action_info.copy_from_src:
                        #     gen_token_mask = 1
                        # else:  # copy
                        #     copy_mask = 1
                        #     copy_pos = action_info.src_token_position
                        #     if token_idx != self.vocab.primitive.unk_id:
                        #         # both copy and generate from vocabulary
                        #         gen_token_mask = 1

                app_rule_idx_row.append(app_rule_idx)
                app_rule_mask_row.append(app_rule_mask)

                token_row.append(token_idx)
                gen_token_mask_row.append(gen_token_mask)
                copy_mask_row.append(copy_mask)

            self.apply_rule_idx_matrix.append(app_rule_idx_row)
            self.apply_rule_mask.append(app_rule_mask_row)

            self.primitive_idx_matrix.append(token_row)
            self.gen_token_mask.append(gen_token_mask_row)

            self.primitive_copy_mask.append(copy_mask_row)

        T = torch.cuda if self.cuda else torch
        self.apply_rule_idx_matrix = Variable(T.LongTensor(self.apply_rule_idx_matrix))
        self.apply_rule_mask = Variable(T.FloatTensor(self.apply_rule_mask))
        self.primitive_idx_matrix = Variable(T.LongTensor(self.primitive_idx_matrix))
        self.gen_token_mask = Variable(T.FloatTensor(self.gen_token_mask))
        self.primitive_copy_mask = Variable(T.FloatTensor(self.primitive_copy_mask))
        self.primitive_copy_token_idx_mask = Variable(torch.from_numpy(self.primitive_copy_token_idx_mask))
        if self.cuda: self.primitive_copy_token_idx_mask = self.primitive_copy_token_idx_mask.cuda()

    @property
    def primitive_mask(self):
        return 1. - torch.eq(self.gen_token_mask + self.primitive_copy_mask, 0).float()

    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          cuda=self.cuda)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)

    @cached_property
    def token_pos_list(self):
        # (batch_size, src_token_pos, unique_src_token_num)

        batch_src_token_to_pos_map = []
        for e_id, e in enumerate(self.examples):
            aggregated_primitive_tokens = OrderedDict()
            for token_pos, token in enumerate(e.src_sent):
                aggregated_primitive_tokens.setdefault(token, []).append(token_pos)


