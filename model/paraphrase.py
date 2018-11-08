# coding=utf-8
import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from components.reranker import RerankingFeature
from model import nn_utils
from model.decomposable_attention_model import DecomposableAttentionModel


class ParaphraseIdentificationModel(nn.Module, RerankingFeature):
    def __init__(self, args, vocab, transition_system):
        super(ParaphraseIdentificationModel, self).__init__()
        self.pi_model = DecomposableAttentionModel(src_vocab=vocab.code, tgt_vocab=vocab.source,
                                                   embed_size=args.embed_size,
                                                   dropout=args.dropout,
                                                   cuda=args.cuda)

        self.vocab = vocab
        self.args = args
        self.transition_system = transition_system

    @property
    def feature_name(self):
        return 'paraphrase_score'

    @property
    def is_batched(self):
        return True

    def _score(self, src_codes, tgt_nls):
        """score examples sorted by code length"""
        args = self.args

        src_code_var = nn_utils.to_input_variable(src_codes, self.vocab.code, cuda=args.cuda).t()
        tgt_nl_var = nn_utils.to_input_variable(tgt_nls, self.vocab.source, cuda=args.cuda).t()

        src_code_mask = nn_utils.length_array_to_mask_tensor([len(x) for x in src_codes], cuda=args.cuda, valid_entry_has_mask_one=True).float()
        tgt_nl_mask = nn_utils.length_array_to_mask_tensor([len(x) for x in tgt_nls], cuda=args.cuda, valid_entry_has_mask_one=True).float()

        scores = self.pi_model(src_code_var, tgt_nl_var, src_code_mask, tgt_nl_mask)

        return scores

    def forward(self, examples):
        tokenized_codes = [self.tokenize_code(e.tgt_code) for e in examples]
        tgt_nls = [e.src_sent for e in examples]

        scores = self._score(tokenized_codes, tgt_nls)

        return scores

    def score(self, examples):
        return self.forward(examples)[:, 0]

    def tokenize_code(self, code):
        return self.transition_system.tokenize_code(code, mode='decoder')

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

        model = ParaphraseIdentificationModel(decoder_params['args'], decoder_params['vocab'], decoder_params['transition_system'])
        model.load_state_dict(decoder_params['state_dict'])

        if cuda: model = model.cuda()
        model.eval()

        return model
