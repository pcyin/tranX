# coding=utf-8
from __future__ import print_function

import os
from itertools import chain
from six.moves import range

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from common.registerable import Registrable
from common.savable import Savable
from components.reranker import RerankingFeature
from model.pointer_net import PointerNet
from model.seq2seq import Seq2SeqModel
from model import nn_utils
from model.seq2seq_copy import Seq2SeqWithCopy


@Registrable.register('reconstructor')
class Reconstructor(nn.Module, RerankingFeature, Savable):
    def __init__(self, args, vocab, transition_system):
        super(Reconstructor, self).__init__()
        if args.no_copy:
            self.seq2seq = Seq2SeqModel(src_vocab=vocab.code, tgt_vocab=vocab.source,
                                        embed_size=args.embed_size, hidden_size=args.hidden_size,
                                        dropout=args.dropout,
                                        label_smoothing=args.src_token_label_smoothing,
                                        cuda=args.cuda)
        else:
            self.seq2seq = Seq2SeqWithCopy(src_vocab=vocab.code, tgt_vocab=vocab.source,
                                           embed_size=args.embed_size, hidden_size=args.hidden_size,
                                           dropout=args.dropout,
                                           cuda=args.cuda)

        self.vocab = vocab
        self.args = args
        self.transition_system = transition_system

    @property
    def feature_name(self):
        return 'reconstructor'

    @property
    def is_batched(self):
        return True

    def _score(self, src_codes, tgt_nls):
        """score examples sorted by code length"""
        args = self.args

        # src_code = [self.tokenize_code(e.tgt_code) for e in examples]
        # tgt_nl = [e.src_sent for e in examples]

        src_code_var = nn_utils.to_input_variable(src_codes, self.vocab.code, cuda=args.cuda)
        tgt_nl_var = nn_utils.to_input_variable(tgt_nls, self.vocab.source, cuda=args.cuda, append_boundary_sym=True)

        tgt_token_copy_idx_mask, tgt_token_gen_mask = self.get_generate_and_copy_meta_tensor(src_codes, tgt_nls)

        if isinstance(self.seq2seq, Seq2SeqWithCopy):
            scores = self.seq2seq(src_code_var,
                                  [len(c) for c in src_codes],
                                  tgt_nl_var,
                                  tgt_token_copy_idx_mask, tgt_token_gen_mask)
        else:
            scores = self.seq2seq(src_code_var,
                                  [len(c) for c in src_codes],
                                  tgt_nl_var)

        return scores

    def score(self, examples):
        batch_size = len(examples)

        tokenized_codes = [self.tokenize_code(e.tgt_code) for e in examples]

        code_lens = [len(code) for code in tokenized_codes]
        sorted_example_ids = sorted(range(batch_size), key=lambda x: -code_lens[x])

        example_old_pos_map = [-1] * batch_size
        for new_pos, old_pos in enumerate(sorted_example_ids):
            example_old_pos_map[old_pos] = new_pos

        sorted_src_codes = [tokenized_codes[i] for i in sorted_example_ids]
        sorted_tgt_nls = [examples[i].src_sent for i in sorted_example_ids]
        sorted_scores = self._score(sorted_src_codes, sorted_tgt_nls)

        scores = sorted_scores[example_old_pos_map]

        return scores

    def forward(self, examples):
        return self.score(examples)

    def sample(self, code, sample_size=5):
        tokenized_code = self.tokenize_code(code)
        samples = self.seq2seq.sample(tokenized_code, sample_size=sample_size, decode_max_time_step=self.args.decode_max_time_step)

        return samples

    def tokenize_code(self, code):
        return self.transition_system.tokenize_code(code, mode='decoder')

    def get_generate_and_copy_meta_tensor(self, src_codes, tgt_nls):
        tgt_nls = [['<s>'] + x + ['</s>'] for x in tgt_nls]
        max_time_step = max(len(tgt_nl) for tgt_nl in tgt_nls)
        max_src_len = max(len(src_code) for src_code in src_codes)
        batch_size = len(src_codes)

        tgt_token_copy_idx_mask = np.zeros((max_time_step, batch_size, max_src_len), dtype='float32')
        tgt_token_gen_mask = np.zeros((max_time_step, batch_size), dtype='float32')

        for t in range(max_time_step):
            for example_id, (src_code, tgt_nl) in enumerate(zip(src_codes, tgt_nls)):
                copy_pos = copy_mask = gen_mask = 0
                if t < len(tgt_nl):
                    tgt_token = tgt_nl[t]
                    copy_pos_list = [_i for _i, _token in enumerate(src_code) if _token == tgt_token]
                    tgt_token_copy_idx_mask[t, example_id, copy_pos_list] = 1

                    gen_mask = 0
                    # we need to generate this token if (1) it's defined in the dictionary,
                    # or (2) it is an unknown word and not appear in the source side
                    if tgt_token in self.vocab.code:
                        gen_mask = 1
                    elif len(copy_pos_list) == 0:
                        gen_mask = 1

                    tgt_token_gen_mask[t, example_id] = gen_mask

        tgt_token_copy_idx_mask = Variable(torch.from_numpy(tgt_token_copy_idx_mask))
        tgt_token_gen_mask = Variable(torch.from_numpy(tgt_token_gen_mask))
        if self.args.cuda:
            tgt_token_copy_idx_mask = tgt_token_copy_idx_mask.cuda()
            tgt_token_gen_mask = tgt_token_gen_mask.cuda()

        return tgt_token_copy_idx_mask, tgt_token_gen_mask

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict(),
            'transition_system': self.transition_system
        }

        torch.save(params, path)

    @staticmethod
    def load(model_path, cuda=False):
        decoder_params = torch.load(model_path, map_location=lambda storage, loc: storage)
        decoder_params['args'].cuda = cuda

        model = Reconstructor(decoder_params['args'], decoder_params['vocab'], decoder_params['transition_system'])
        model.load_state_dict(decoder_params['state_dict'])

        if cuda: model = model.cuda()
        model.eval()

        return model
