# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import nn_utils
from model.pointer_net import PointerNet
from model.seq2seq import Seq2SeqModel


class Seq2SeqWithCopy(Seq2SeqModel):
    def __init__(self, src_vocab, tgt_vocab, embed_size, hidden_size,
                 ptrnet_hidden_dim,
                 dropout=0.,
                 cuda=False,
                 src_embed_layer=None, tgt_embed_layer=None):

        super(Seq2SeqWithCopy, self).__init__(src_vocab, tgt_vocab,
                                              embed_size, hidden_size,
                                              dropout=dropout,
                                              src_embed_layer=src_embed_layer, tgt_embed_layer=tgt_embed_layer,
                                              cuda=cuda)

        # pointer net to the source
        self.src_pointer_net = PointerNet(src_encoding_size=hidden_size * 2,
                                          query_vec_size=hidden_size,
                                          hidden_dim=ptrnet_hidden_dim)

        self.tgt_token_predictor = nn.Linear(hidden_size, 2)

    def encode(self, src_sents_var, src_sents_len):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
            dec_init_state, dec_init_cell: Variable(batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
        src_token_embed = self.src_embed(src_sents_var)
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)

        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)
        # (batch_size, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        return src_encodings, (last_state, last_cell)

    def decode(self, src_encodings, src_sent_masks, dec_init_vec, tgt_sents_var):
        new_tensor = src_encodings.data.new
        batch_size = src_encodings.size(0)

        h_tm1 = dec_init_vec

        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        # initialize the attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.hidden_size).zero_(), requires_grad=False)

        # (tgt_sent_len, batch_size, embed_size)
        tgt_token_embed = self.tgt_embed(tgt_sents_var)

        att_ves = []
        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(tgt_token_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim
            y_tm1_embed = y_tm1_embed.squeeze(0)
            x = torch.cat([y_tm1_embed, att_tm1], 1)

            (h_t, cell_t), att_t = self.step(x, h_tm1,
                                             src_encodings, src_encodings_att_linear,
                                             src_sent_masks=src_sent_masks)

            att_ves.append(att_t)

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        # (src_sent_len, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        return att_ves

    def forward(self, src_sents_var, src_sents_len, tgt_sents_var, tgt_token_copy_pos, tgt_token_copy_mask, tgt_token_gen_mask):
        """
        compute log p(y|x)

        :param tgt_token_copy_pos: Variable(tgt_action_len, batch_size)
        :param tgt_token_copy_mask: Variable(tgt_action_len, batch_size)
        :return: Variable(batch_size)
        """

        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        # (batch_size, src_sent_len)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, cuda=self.cuda)

        # (tgt_sent_len - 1, batch_size, hidden_size)
        att_vecs = self.decode(src_encodings, src_sent_masks, dec_init_vec, tgt_sents_var)

        # (tgt_sent_len - 1, batch_size, 2)
        tgt_token_predictor = F.softmax(self.tgt_token_predictor(att_vecs), dim=-1)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        token_gen_prob = F.softmax(self.readout(att_vecs), dim=-1)
        # (tgt_sent_len - 1, batch_size, src_sent_len)
        token_copy_prob = self.src_pointer_net(src_encodings, src_sent_masks, att_vecs)

        tgt_token_idx = tgt_sents_var[1:]  # remove leading </s>
        tgt_token_copy_pos = tgt_token_copy_pos[1:]
        tgt_token_gen_mask = tgt_token_gen_mask[1:]
        tgt_token_copy_mask = tgt_token_copy_mask[1:]

        # (tgt_sent_len - 1, batch_size)
        tgt_token_gen_prob = torch.gather(token_gen_prob, dim=2,
                                          index=tgt_token_idx.unsqueeze(2)).squeeze(2) * tgt_token_gen_mask

        # (tgt_sent_len - 1, batch_size)
        tgt_token_copy_prob = torch.gather(token_copy_prob, dim=2, index=tgt_token_copy_pos.unsqueeze(2)).squeeze(2) * tgt_token_copy_mask

        tgt_token_mask = torch.gt(tgt_token_gen_mask + tgt_token_copy_mask, 0.).float()
        tgt_token_prob = torch.log(tgt_token_predictor[:, :, 0] * tgt_token_gen_prob +
                                   tgt_token_predictor[:, :, 1] * tgt_token_copy_prob +
                                   1.e-7 * (1. - tgt_token_mask))
        tgt_token_prob = tgt_token_prob * tgt_token_mask

        # (batch_size)
        scores = tgt_token_prob.sum(dim=0)

        return scores

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_sent_masks)

        att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t
