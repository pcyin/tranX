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
                                          query_vec_size=hidden_size)

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

    def forward(self, src_sents_var, src_sents_len, tgt_sents_var, tgt_token_copy_idx_mask, tgt_token_gen_mask):
        """
        compute log p(y|x)

        :param tgt_token_copy_idx_mask: Variable(tgt_action_len, batch_size, src_seq_len)
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

        tgt_token_idx = tgt_sents_var[1:]  # remove leading <s>
        tgt_token_gen_mask = tgt_token_gen_mask[1:]
        tgt_token_copy_idx_mask = tgt_token_copy_idx_mask[1:]

        # (tgt_sent_len - 1, batch_size)
        tgt_token_gen_prob = torch.gather(token_gen_prob, dim=2,
                                          index=tgt_token_idx.unsqueeze(2)).squeeze(2) * tgt_token_gen_mask

        # (tgt_sent_len - 1, batch_size)
        tgt_token_copy_prob = torch.sum(token_copy_prob * tgt_token_copy_idx_mask, dim=-1)
        # tgt_token_copy_prob = torch.gather(token_copy_prob, dim=2, index=tgt_token_copy_pos.unsqueeze(2)).squeeze(2) * tgt_token_copy_mask

        tgt_token_mask = torch.gt(tgt_token_gen_mask + tgt_token_copy_idx_mask.sum(dim=-1), 0.).float()
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

    def sample(self, src_sent, sample_size, decode_max_time_step, cuda=False, mode='sample'):
        new_float_tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        new_long_tensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        src_sent_var = nn_utils.to_input_variable([src_sent], self.src_vocab,
                                                  cuda=cuda, training=False)

        # analyze which tokens can be copied from the source
        src_token_tgt_vocab_ids = [self.tgt_vocab[token] for token in src_sent]
        src_unk_pos_list = [pos for pos, token_id in enumerate(src_token_tgt_vocab_ids) if
                            token_id == self.tgt_vocab.unk_id]
        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_tgt_vocab_ids):
            if tid in token_set:
                src_token_tgt_vocab_ids[i] = -1
            else:
                token_set.add(tid)

        src_encodings, (last_state, last_cell) = self.encode(src_sent_var, [len(src_sent)])
        h_tm1 = self.init_decoder_state(last_state, last_cell)

        # (batch_size, 1, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        t = 0
        eos_id = self.tgt_vocab['</s>']

        completed_hypotheses = []
        completed_hypothesis_scores = []

        if mode == 'beam_search':
            hypotheses = [['<s>']]
            hypotheses_word_ids = [[self.tgt_vocab['<s>']]]
        else:
            hypotheses = [['<s>'] for _ in range(sample_size)]
            hypotheses_word_ids = [[self.tgt_vocab['<s>']] for _ in range(sample_size)]

        att_tm1 = Variable(new_float_tensor(len(hypotheses), self.hidden_size).zero_(), volatile=True)
        hyp_scores = Variable(new_float_tensor(len(hypotheses)).zero_(), volatile=True)

        while len(completed_hypotheses) < sample_size and t < decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                                src_encodings_att_linear.size(1),
                                                                                src_encodings_att_linear.size(2))

            y_tm1 = Variable(new_long_tensor([hyp[-1] for hyp in hypotheses_word_ids]), volatile=True)
            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            (h_t, cell_t), att_t = self.step(x, h_tm1,
                                             expanded_src_encodings, expanded_src_encodings_att_linear)

            # (batch_size, 2)
            tgt_token_predictor = F.softmax(self.tgt_token_predictor(att_t), dim=-1)

            # (batch_size, tgt_vocab_size)
            token_gen_prob = F.softmax(self.readout(att_t), dim=-1)

            # (batch_size, src_sent_len)
            token_copy_prob = self.src_pointer_net(src_encodings, src_token_mask=None, query_vec=att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, tgt_vocab_size)
            token_gen_prob = tgt_token_predictor[:, 0].unsqueeze(1) * token_gen_prob

            for token_pos, token_vocab_id in enumerate(src_token_tgt_vocab_ids):
                if token_vocab_id != -1 and token_vocab_id != self.tgt_vocab.unk_id:
                    p_copy = tgt_token_predictor[:, 1] * token_copy_prob[:, token_pos]
                    token_gen_prob[:, token_vocab_id] = token_gen_prob[:, token_vocab_id] + p_copy

            # second, add the probability of copying the most probable unk word
            gentoken_new_hyp_unks = []
            if src_unk_pos_list:
                for hyp_id in range(hyp_num):
                    unk_pos = token_copy_prob[hyp_id][src_unk_pos_list].data.cpu().numpy().argmax()
                    unk_pos = src_unk_pos_list[unk_pos]
                    token = src_sent[unk_pos]
                    gentoken_new_hyp_unks.append(token)

                    unk_copy_score = tgt_token_predictor[hyp_id, 1] * token_copy_prob[hyp_id, unk_pos]
                    token_gen_prob[hyp_id, self.tgt_vocab.unk_id] = unk_copy_score

            live_hyp_num = sample_size - len(completed_hypotheses)

            if mode == 'beam_search':
                log_token_gen_prob = torch.log(token_gen_prob)
                new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(token_gen_prob) + log_token_gen_prob).view(-1)
                top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
                prev_hyp_ids = (top_new_hyp_pos / len(self.tgt_vocab)).cpu().data
                word_ids = (top_new_hyp_pos % len(self.tgt_vocab)).cpu().data
                top_new_hyp_scores = top_new_hyp_scores.cpu().data
            else:
                word_ids = torch.multinomial(token_gen_prob, num_samples=1)
                prev_hyp_ids = range(live_hyp_num)
                top_new_hyp_scores = hyp_scores + torch.log(torch.gather(token_gen_prob, dim=1, index=word_ids)).squeeze(1)
                top_new_hyp_scores = top_new_hyp_scores.cpu().data
                word_ids = word_ids.view(-1).cpu().data

            new_hypotheses = []
            new_hypotheses_word_ids = []
            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids, word_ids, top_new_hyp_scores):
                if word_id == eos_id:
                    hyp_tgt_words = hypotheses[prev_hyp_id][1:]
                    completed_hypotheses.append(hyp_tgt_words)  # remove <s> and </s> in completed hypothesis
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    if word_id == self.tgt_vocab.unk_id:
                        if gentoken_new_hyp_unks: word = gentoken_new_hyp_unks[prev_hyp_id]
                        else: word = self.tgt_vocab.id2word[self.tgt_vocab.unk_id]
                    else:
                        word = self.tgt_vocab.id2word[word_id]

                    hyp_tgt_words = hypotheses[prev_hyp_id] + [word]
                    new_hypotheses.append(hyp_tgt_words)
                    new_hypotheses_word_ids.append(hypotheses_word_ids[prev_hyp_id] + [word_id])
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == sample_size:
                break

            live_hyp_ids = new_long_tensor(live_hyp_ids)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = Variable(new_float_tensor(new_hyp_scores), volatile=True)  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
            hypotheses_word_ids = new_hypotheses_word_ids

        return completed_hypotheses
