# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import nn_utils


class Seq2SeqModel(nn.Module):
    """
    a standard seq2seq model
    """
    def __init__(self, src_vocab, tgt_vocab, embed_size, hidden_size,
                 decoder_word_dropout=0., dropout=0.,
                 label_smoothing=0.,
                 cuda=False,
                 src_embed_layer=None, tgt_embed_layer=None):
        super(Seq2SeqModel, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size

        if src_embed_layer:
            self.src_embed = src_embed_layer
        else:
            self.src_embed = nn.Embedding(len(src_vocab), embed_size)

        if tgt_embed_layer:
            self.tgt_embed = tgt_embed_layer
        else:
            self.tgt_embed = nn.Embedding(len(tgt_vocab), embed_size)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.decoder_lstm = nn.LSTMCell(embed_size + hidden_size, hidden_size)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(hidden_size * 2, hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(tgt_vocab), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(dropout)
        self.decoder_word_dropout = decoder_word_dropout

        # label smoothing
        self.label_smoothing = label_smoothing
        if label_smoothing:
            self.label_smoothing_layer = nn_utils.LabelSmoothing(label_smoothing, len(tgt_vocab), ignore_indices=[0])

        self.cuda = cuda

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

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        return src_encodings, (last_state, last_cell)

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        dec_init_cell = self.decoder_cell_init(enc_last_cell)
        dec_init_state = F.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

    def decode(self, src_encodings, src_sents_len, dec_init_vec, tgt_sents_var):
        """
        compute the final softmax layer at each decoding step
        :param src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
        :param src_sents_len: list[int]
        :param dec_init_vec: tuple((batch_size, hidden_size))
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            scores: Variable(src_sent_len, batch_size, src_vocab_size)
        """
        new_tensor = src_encodings.data.new
        batch_size = src_encodings.size(1)

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        # initialize the attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.hidden_size).zero_(), requires_grad=False)
        # (batch_size, src_sent_len)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, cuda=self.cuda)

        # (tgt_sent_len, batch_size, embed_size)
        tgt_token_embed = self.tgt_embed(tgt_sents_var)

        scores = []
        # start from `<s>`, until y_{T-1}
        for t, y_tm1_embed in list(enumerate(tgt_token_embed.split(split_size=1)))[:-1]:
            # input feeding: concate y_tm1 and previous attentional vector
            # split() keeps the first dim
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if t > 0 and self.decoder_word_dropout:
                # (batch_size)
                y_tm1_mask = Variable(torch.bernoulli(new_tensor(batch_size).fill_(1 - self.decoder_word_dropout)))
                y_tm1_embed = y_tm1_embed * y_tm1_mask.unsqueeze(1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            (h_t, cell_t), att_t, score_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks)

            scores.append(score_t)

            att_tm1 = att_t
            h_tm1 = (h_t, cell_t)

        # (src_sent_len, batch_size, tgt_vocab_size)
        scores = torch.stack(scores)

        return scores

    def score_decoding_results(self, scores, tgt_sents_var):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var: Variable(src_sent_len, batch_size)
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """
        batch_size = scores.size(1)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        log_scores = F.log_softmax(scores, dim=-1)
        tgt_sents_var_sos_omitted = tgt_sents_var[1:]   # remove leading <s> in tgt sent, which is not used as the target

        if self.training and self.label_smoothing:
            # (tgt_sent_len, batch_size)
            tgt_sent_log_scores = -self.label_smoothing_layer(log_scores, tgt_sents_var_sos_omitted)
        else:
            # (tgt_sent_len, batch_size)
            tgt_sent_log_scores = torch.gather(log_scores, -1, tgt_sents_var_sos_omitted.unsqueeze(-1)).squeeze(-1)

        tgt_sent_log_scores = tgt_sent_log_scores * (1. - torch.eq(tgt_sents_var_sos_omitted, 0).float())  # 0 is pad

        # (batch_size)
        tgt_sent_log_scores = tgt_sent_log_scores.sum(dim=0)

        return tgt_sent_log_scores

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = Seq2SeqModel.dot_prod_attention(h_t,
                                                         src_encodings, src_encodings_att_linear,
                                                         mask=src_sent_masks)

        att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        # (batch_size, tgt_vocab_size)
        score_t = self.readout(att_t)  # E.q. (6)

        return (h_t, cell_t), att_t, score_t

    def forward(self, src_sents_var, src_sents_len, tgt_sents_var):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)
        """

        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        tgt_token_logits = self.decode(src_encodings, src_sents_len, dec_init_vec, tgt_sents_var)
        tgt_sent_log_scores = self.score_decoding_results(tgt_token_logits, tgt_sents_var)

        return tgt_sent_log_scores

    @staticmethod
    def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask is not None:
            att_weight.data.masked_fill_(mask, -float('inf'))
        att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def sample(self, src_sents, sample_size):
        src_sents_len = [len(src_sent) for src_sent in src_sents]
        # Variable: (src_sent_len, batch_size)
        src_sents_var = nn_utils.to_input_variable(src_sents, self.vocab.src,
                                                   cuda=self.cuda, training=False)
        return self.sample_from_variable(src_sents_var, src_sents_len, sample_size)

    def sample_from_src_variable(self, src_sents_var, src_sents_len, sample_size):
        # (batch_size * sample_size, src_sent_len)
        src_sent_masks = nn_utils.length_array_to_mask_tensor(
            list(chain.from_iterable([l] * sample_size for l in src_sents_len)),
            cuda=self.cuda)

        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        return self.sample_from_src_encoding(src_encodings, dec_init_vec, sample_size, src_sent_masks)

    def sample_from_src_encoding(self, src_encodings, dec_init_vec, sample_size, decode_max_time_step, src_sent_masks=None):
        src_sents_num = dec_init_vec[0].size(0) * sample_size

        # tensor constructors
        new_float_tensor = src_encodings.data.new
        if self.cuda:
            new_long_tensor = torch.cuda.LongTensor
        else:
            new_long_tensor = torch.LongTensor

        # (batch_size * sample_size, hidden_size * 2)
        dec_init_state, dec_init_cell = dec_init_vec
        dec_init_state = dec_init_state.repeat(1, sample_size).view(src_sents_num, -1)
        dec_init_cell = dec_init_cell.repeat(1, sample_size).view(src_sents_num, -1)
        h_tm1 = (dec_init_state, dec_init_cell)

        # (query_len, batch_size * sample_size, hidden_size * 2)
        src_encodings = src_encodings.repeat(1, 1, sample_size).view(src_encodings.size(0), src_sents_num,
                                                                     src_encodings.size(2))
        # (batch_size * sample_size, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        att_tm1 = Variable(new_float_tensor(src_sents_num, self.hidden_size).zero_())
        y_0 = Variable(new_long_tensor([self.tgt_vocab['<s>'] for _ in xrange(src_sents_num)]))

        eos_wid = self.tgt_vocab['</s>']

        samples_var = [y_0]
        samples = [['<s>'] for _ in xrange(src_sents_num)]
        samples_len = [0] * src_sents_num
        sample_scores = []
        t = 0
        while t < decode_max_time_step:
            t += 1

            # (sample_size)
            y_tm1 = samples_var[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (batch_size * sample_size, hidden_size)
            (h_t, cell_t), att_t, score_t = self.step(x, h_tm1,
                                                      src_encodings, src_encodings_att_linear,
                                                      src_sent_masks=src_sent_masks)

            # (batch_size * sample_size, tgt_vocab_size)
            p_t = F.softmax(score_t)
            # (batch_size * sample_size, 1)
            y_t = torch.multinomial(p_t, num_samples=1).detach()
            # (batch_size * sample_size, 1)
            p_y_t = torch.gather(p_t, 1, y_t)
            log_y_t = torch.log(p_y_t).squeeze(1)

            y_t = y_t.squeeze(1)

            # generate loss mask
            mask_t = []
            is_valid_mask = False
            for sample_id, y in enumerate(y_t.cpu().data):
                if samples_len[sample_id] == 0:
                    mask_t.append(1.)
                    word = self.tgt_vocab.id2word[y]
                    samples[sample_id].append(word)
                    if y == eos_wid:
                        samples_len[sample_id] = t + 1
                else:
                    mask_t.append(0.)
                    is_valid_mask = True

            if is_valid_mask:
                mask_t = Variable(new_float_tensor(mask_t))
                log_y_t = log_y_t * mask_t

            samples_var.append(y_t)
            sample_scores.append(log_y_t)

            if all(l > 0 for l in samples_len):
                break

            att_tm1 = att_t
            h_tm1 = h_t, cell_t

        # (max_sample_len, batch_size * sample_size)
        sample_scores = torch.stack(sample_scores)

        # finally, let's remove <s> and </s> from the samples!
        samples = [sample[1:-1] for sample in samples]

        return samples, sample_scores

    def beam_search(self, src_sents, decode_max_time_step, beam_size=5, to_word=True):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        src_sents_var = nn_utils.to_input_variable(src_sents, self.src_vocab,
                                                   cuda=self.cuda, training=False, append_boundary_sym=False)

        #TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        src_encodings, (last_state, last_cell) = self.encode(src_sents_var, [len(src_sents[0])])
        # (1, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        h_tm1 = self.init_decoder_state(last_state, last_cell)

        # tensor constructors
        new_float_tensor = src_encodings.data.new
        if self.cuda:
            new_long_tensor = torch.cuda.LongTensor
        else:
            new_long_tensor = torch.LongTensor

        att_tm1 = Variable(torch.zeros(1, self.hidden_size), volatile=True)
        hyp_scores = Variable(torch.zeros(1), volatile=True)
        if self.cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        eos_id = self.tgt_vocab['</s>']
        bos_id = self.tgt_vocab['<s>']
        tgt_vocab_size = len(self.tgt_vocab)

        hypotheses = [[bos_id]]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))

            y_tm1 = Variable(new_long_tensor([hyp[-1] for hyp in hypotheses]), volatile=True)
            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            (h_t, cell_t), att_t, score_t = self.step(x, h_tm1,
                                                      expanded_src_encodings, expanded_src_encodings_att_linear,
                                                      src_sent_masks=None)

            p_t = F.log_softmax(score_t)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos / tgt_vocab_size
            word_ids = top_new_hyp_pos % tgt_vocab_size

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data, top_new_hyp_scores.cpu().data):
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if word_id == eos_id:
                    completed_hypotheses.append(hyp_tgt_words[1:-1])  # remove <s> and </s> in completed hypothesis
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    new_hypotheses.append(hyp_tgt_words)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = new_long_tensor(live_hyp_ids)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = Variable(new_float_tensor(new_hyp_scores), volatile=True)  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0][1:-1]]  # remove <s> and </s> in completed hypothesis
            completed_hypothesis_scores = [0.0]

        if to_word:
            for i, hyp in enumerate(completed_hypotheses):
                completed_hypotheses[i] = [self.tgt_vocab.id2word[w] for w in hyp]

        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores), key=lambda x: x[1], reverse=True)

        return [hyp for hyp, score in ranked_hypotheses]

