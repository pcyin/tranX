# coding=utf-8
from __future__ import print_function

import os
from six.moves import xrange as range
import math
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from asdl.hypothesis import Hypothesis, GenTokenAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, Action
from components.decode_hypothesis import DecodeHypothesis
from components.action_info import ActionInfo
from components.dataset import Batch
from common.utils import update_args, init_arg_parser
from model import nn_utils
from model.attention_util import AttentionUtil
from model.nn_utils import LabelSmoothing
from model.pointer_net import PointerNet


class Parser(nn.Module):
    """Implementation of a semantic parser

    The parser translates a natural language utterance into an AST defined under
    the ASDL specification, using the transition system described in https://arxiv.org/abs/1810.02720
    """
    def __init__(self, args, vocab, transition_system):
        super(Parser, self).__init__()

        self.args = args
        self.vocab = vocab

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        # Embedding layers

        # source token embedding
        self.src_embed = nn.Embedding(len(vocab.source), args.embed_size)

        # embedding table of ASDL production rules (constructors), one for each ApplyConstructor action,
        # the last entry is the embedding for Reduce action
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)

        # embedding table for target primitive tokens
        self.primitive_embed = nn.Embedding(len(vocab.primitive), args.action_embed_size)

        # embedding table for ASDL fields in constructors
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)

        # embedding table for ASDL types
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        nn.init.xavier_normal(self.src_embed.weight.data)
        nn.init.xavier_normal(self.production_embed.weight.data)
        nn.init.xavier_normal(self.primitive_embed.weight.data)
        nn.init.xavier_normal(self.field_embed.weight.data)
        nn.init.xavier_normal(self.type_embed.weight.data)

        # LSTMs
        if args.lstm == 'lstm':
            self.encoder_lstm = nn.LSTM(args.embed_size, int(args.hidden_size / 2), bidirectional=True)

            input_dim = args.action_embed_size  # previous action
            # frontier info
            input_dim += args.action_embed_size * (not args.no_parent_production_embed)
            input_dim += args.field_embed_size * (not args.no_parent_field_embed)
            input_dim += args.type_embed_size * (not args.no_parent_field_type_embed)
            input_dim += args.hidden_size * (not args.no_parent_state)

            input_dim += args.att_vec_size * (not args.no_input_feed)  # input feeding

            self.decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)
        elif args.lstm == 'parent_feed':
            self.encoder_lstm = nn.LSTM(args.embed_size, int(args.hidden_size / 2), bidirectional=True)
            from .lstm import ParentFeedingLSTMCell

            input_dim = args.action_embed_size  # previous action
            # frontier info
            input_dim += args.action_embed_size * (not args.no_parent_production_embed)
            input_dim += args.field_embed_size * (not args.no_parent_field_embed)
            input_dim += args.type_embed_size * (not args.no_parent_field_type_embed)
            input_dim += args.att_vec_size * (not args.no_input_feed)  # input feeding

            self.decoder_lstm = ParentFeedingLSTMCell(input_dim, args.hidden_size)
        else:
            raise ValueError('Unknown LSTM type %s' % args.lstm)

        if args.no_copy is False:
            # pointer net for copying tokens from source side
            self.src_pointer_net = PointerNet(query_vec_size=args.att_vec_size, src_encoding_size=args.hidden_size)

            # given the decoder's hidden state, predict whether to copy or generate a target primitive token
            # output: [p(gen(token)) | s_t, p(copy(token)) | s_t]

            self.primitive_predictor = nn.Linear(args.att_vec_size, 2)

        if args.primitive_token_label_smoothing:
            self.label_smoothing = LabelSmoothing(args.primitive_token_label_smoothing, len(self.vocab.primitive), ignore_indices=[0, 1, 2])

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's hidden space

        self.att_src_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)

        self.att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        # bias for predicting ApplyConstructor and GenToken actions
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(transition_system.grammar) + 1).zero_())
        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab.primitive)).zero_())

        if args.no_query_vec_to_action_map:
            # if there is no additional linear layer between the attentional vector (i.e., the query vector)
            # and the final softmax layer over target actions, we use the attentional vector to compute action
            # probabilities

            assert args.att_vec_size == args.action_embed_size
            self.production_readout = lambda q: F.linear(q, self.production_embed.weight, self.production_readout_b)
            self.tgt_token_readout = lambda q: F.linear(q, self.primitive_embed.weight, self.tgt_token_readout_b)
        else:
            # by default, we feed the attentional vector (i.e., the query vector) into a linear layer without bias, and
            # compute action probabilities by dot-producting the resulting vector and (GenToken, ApplyConstructor) action embeddings
            # i.e., p(action) = query_vec^T \cdot W \cdot embedding

            self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            if args.query_vec_to_action_diff_map:
                # use different linear transformations for GenToken and ApplyConstructor actions
                self.query_vec_to_primitive_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            else:
                self.query_vec_to_primitive_embed = self.query_vec_to_action_embed

            self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity

            self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                         self.production_embed.weight, self.production_readout_b)
            self.tgt_token_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                        self.primitive_embed.weight, self.tgt_token_readout_b)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def encode(self, src_sents_var, src_sents_len):
        """Encode the input natural language utterance

        Args:
            src_sents_var: a variable of shape (src_sent_len, batch_size), representing word ids of the input
            src_sents_len: a list of lengths of input source sentences, sorted by descending order

        Returns:
            src_encodings: source encodings of shape (batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: the last hidden state and cell state of the encoder,
                                   of shape (batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
        # apply word dropout
        if self.training and self.args.word_dropout:
            mask = Variable(self.new_tensor(src_sents_var.size()).fill_(1. - self.args.word_dropout).bernoulli().long())
            src_sents_var = src_sents_var * mask + (1 - mask) * self.vocab.source.unk_id

        src_token_embed = self.src_embed(src_sents_var)
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)

        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        src_encodings = src_encodings.permute(1, 0, 2)

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        return src_encodings, (last_state, last_cell)

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        """Compute the initial decoder hidden state and cell state"""

        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = F.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

    def score(self, examples, return_encode_state=False):
        """Given a list of examples, compute the log-likelihood of generating the target AST

        Args:
            examples: a batch of examples
            return_encode_state: return encoding states of input utterances
        output: score for each training example: Variable(batch_size)
        """

        batch = Batch(examples, self.grammar, self.vocab, copy=self.args.no_copy is False, cuda=self.args.cuda)

        # src_encodings: (batch_size, src_sent_len, hidden_size * 2)
        # (last_state, last_cell, dec_init_vec): (batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        # query vectors are sufficient statistics used to compute action probabilities
        # query_vectors: (tgt_action_len, batch_size, hidden_size)

        # if use supervised attention
        if self.args.sup_attention:
            query_vectors, att_prob = self.decode(batch, src_encodings, dec_init_vec)
        else:
            query_vectors = self.decode(batch, src_encodings, dec_init_vec)

        # ApplyRule (i.e., ApplyConstructor) action probabilities
        # (tgt_action_len, batch_size, grammar_size)
        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)

        # probabilities of target (gold-standard) ApplyRule actions
        # (tgt_action_len, batch_size)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
                                           index=batch.apply_rule_idx_matrix.unsqueeze(2)).squeeze(2)

        #### compute generation and copying probabilities

        # (tgt_action_len, batch_size, primitive_vocab_size)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # (tgt_action_len, batch_size)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(2)).squeeze(2)

        if self.args.no_copy:
            # mask positions in action_prob that are not used

            if self.training and self.args.primitive_token_label_smoothing:
                # (tgt_action_len, batch_size)
                # this is actually the negative KL divergence size we will flip the sign later
                # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
                #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))

                tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                    gen_from_vocab_prob.log(),
                    batch.primitive_idx_matrix)
            else:
                tgt_primitive_gen_from_vocab_log_prob = tgt_primitive_gen_from_vocab_prob.log()

            # (tgt_action_len, batch_size)
            action_prob = tgt_apply_rule_prob.log() * batch.apply_rule_mask + \
                          tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask
        else:
            # binary gating probabilities between generating or copying a primitive token
            # (tgt_action_len, batch_size, 2)
            primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)

            # pointer network copying scores over source tokens
            # (tgt_action_len, batch_size, src_sent_len)
            primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)

            # marginalize over the copy probabilities of tokens that are same
            # (tgt_action_len, batch_size)
            tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)

            # mask positions in action_prob that are not used
            # (tgt_action_len, batch_size)
            action_mask_pad = torch.eq(batch.apply_rule_mask + batch.gen_token_mask + batch.primitive_copy_mask, 0.)
            action_mask = 1. - action_mask_pad.float()

            # (tgt_action_len, batch_size)
            action_prob = tgt_apply_rule_prob * batch.apply_rule_mask + \
                          primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask + \
                          primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask

            # avoid nan in log
            action_prob.data.masked_fill_(action_mask_pad.data, 1.e-7)

            action_prob = action_prob.log() * action_mask

        scores = torch.sum(action_prob, dim=0)

        returns = [scores]
        if self.args.sup_attention:
            returns.append(att_prob)
        if return_encode_state: returns.append(last_state)

        return returns

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_token_mask=None, return_att_weight=False):
        """Perform a single time-step of computation in decoder LSTM

        Args:
            x: variable of shape (batch_size, hidden_size), input
            h_tm1: Tuple[Variable(batch_size, hidden_size), Variable(batch_size, hidden_size)], previous
                   hidden and cell states
            src_encodings: variable of shape (batch_size, src_sent_len, hidden_size * 2), encodings of source utterances
            src_encodings_att_linear: linearly transformed source encodings
            src_token_mask: mask over source tokens (Note: unused entries are masked to **one**)
            return_att_weight: return attention weights

        Returns:
            The new LSTM hidden state and cell state
        """

        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else: return (h_t, cell_t), att_t

    def decode(self, batch, src_encodings, dec_init_vec):
        """Given a batch of examples and their encodings of input utterances,
        compute query vectors at each decoding time step, which are used to compute
        action probabilities

        Args:
            batch: a `Batch` object storing input examples
            src_encodings: variable of shape (batch_size, src_sent_len, hidden_size * 2), encodings of source utterances
            dec_init_vec: a tuple of variables representing initial decoder states

        Returns:
            Query vectors, a variable of shape (tgt_action_len, batch_size, hidden_size)
            Also return the attention weights over candidate tokens if using supervised attention
        """

        batch_size = len(batch)
        args = self.args

        if args.lstm == 'parent_feed':
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    Variable(self.new_tensor(batch_size, args.hidden_size).zero_()), \
                    Variable(self.new_tensor(batch_size, args.hidden_size).zero_())
        else:
            h_tm1 = dec_init_vec

        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        att_vecs = []
        history_states = []
        att_probs = []
        att_weights = []

        for t in range(batch.max_action_num):
            # the input to the decoder LSTM is a concatenation of multiple signals
            # [
            #   embedding of previous action -> `a_tm1_embed`,
            #   previous attentional vector -> `att_tm1`,
            #   embedding of the current frontier (parent) constructor (rule) -> `parent_production_embed`,
            #   embedding of the frontier (parent) field -> `parent_field_embed`,
            #   embedding of the ASDL type of the frontier field -> `parent_field_type_embed`,
            #   LSTM state of the parent action -> `parent_states`
            # ]

            if t == 0:
                x = Variable(self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_(), requires_grad=False)

                # initialize using the root type embedding
                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.att_vec_size * (not args.no_input_feed)
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[:, offset: offset + args.type_embed_size] = self.type_embed(Variable(self.new_long_tensor(
                        [self.grammar.type2id[self.grammar.root_type] for e in batch.examples])))
            else:
                a_tm1_embeds = []
                for example in batch.examples:
                    # action t - 1
                    if t < len(example.tgt_actions):
                        a_tm1 = example.tgt_actions[t - 1]
                        if isinstance(a_tm1.action, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.action.production]]
                        elif isinstance(a_tm1.action, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.action.token]]
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if args.no_input_feed is False:
                    inputs.append(att_tm1)
                if args.no_parent_production_embed is False:
                    parent_production_embed = self.production_embed(batch.get_frontier_prod_idx(t))
                    inputs.append(parent_production_embed)
                if args.no_parent_field_embed is False:
                    parent_field_embed = self.field_embed(batch.get_frontier_field_idx(t))
                    inputs.append(parent_field_embed)
                if args.no_parent_field_type_embed is False:
                    parent_field_type_embed = self.type_embed(batch.get_frontier_field_type_idx(t))
                    inputs.append(parent_field_type_embed)

                # append history states
                actions_t = [e.tgt_actions[t] if t < len(e.tgt_actions) else None for e in batch.examples]
                if args.no_parent_state is False:
                    parent_states = torch.stack([history_states[p_t][0][batch_id]
                                                 for batch_id, p_t in
                                                 enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])

                    parent_cells = torch.stack([history_states[p_t][1][batch_id]
                                                for batch_id, p_t in
                                                enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])

                    if args.lstm == 'parent_feed':
                        h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)
                    else:
                        inputs.append(parent_states)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, src_encodings,
                                                         src_encodings_att_linear,
                                                         src_token_mask=batch.src_token_mask,
                                                         return_att_weight=True)

            # if use supervised attention
            if args.sup_attention:
                for e_id, example in enumerate(batch.examples):
                    if t < len(example.tgt_actions):
                        action_t = example.tgt_actions[t].action
                        cand_src_tokens = AttentionUtil.get_candidate_tokens_to_attend(example.src_sent, action_t)
                        if cand_src_tokens:
                            att_prob = [att_weight[e_id, token_id] for token_id in cand_src_tokens]
                            if len(att_prob) > 1: att_prob = torch.cat(att_prob).sum()
                            else: att_prob = att_prob[0]
                            att_probs.append(att_prob)

            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)
            att_weights.append(att_weight)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        att_vecs = torch.stack(att_vecs, dim=0)
        if args.sup_attention:
            return att_vecs, att_probs
        else: return att_vecs

    def parse(self, src_sent, context=None, beam_size=5, debug=False):
        """Perform beam search to infer the target AST given a source utterance

        Args:
            src_sent: list of source utterance tokens
            context: other context used for prediction
            beam_size: beam size

        Returns:
            A list of `DecodeHypothesis`, each representing an AST
        """

        args = self.args
        primitive_vocab = self.vocab.primitive
        T = torch.cuda if args.cuda else torch

        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, cuda=args.cuda, training=False)

        # Variable(1, src_sent_len, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encode(src_sent_var, [len(src_sent)])
        # (1, src_sent_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        if args.lstm == 'parent_feed':
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    Variable(self.new_tensor(args.hidden_size).zero_()), \
                    Variable(self.new_tensor(args.hidden_size).zero_())
        else:
            h_tm1 = dec_init_vec

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        hyp_scores = Variable(self.new_tensor([0.]), volatile=True)

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        t = 0
        hypotheses = [DecodeHypothesis()]
        hyp_states = [[]]
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))

            if t == 0:
                x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).zero_(), volatile=True)
                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.att_vec_size * (not args.no_input_feed)
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[0, offset: offset + args.type_embed_size] = \
                        self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]

                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                        elif isinstance(a_tm1, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]

                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if args.no_input_feed is False:
                    inputs.append(att_tm1)
                if args.no_parent_production_embed is False:
                    # frontier production
                    frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                    frontier_prod_embeds = self.production_embed(Variable(self.new_long_tensor(
                        [self.grammar.prod2id[prod] for prod in frontier_prods])))
                    inputs.append(frontier_prod_embeds)
                if args.no_parent_field_embed is False:
                    # frontier field
                    frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                    frontier_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                        self.grammar.field2id[field] for field in frontier_fields])))

                    inputs.append(frontier_field_embeds)
                if args.no_parent_field_type_embed is False:
                    # frontier field type
                    frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                    frontier_field_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                        self.grammar.type2id[type] for type in frontier_field_types])))
                    inputs.append(frontier_field_type_embeds)

                # parent states
                if args.no_parent_state is False:
                    p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                    parent_states = torch.stack([hyp_states[hyp_id][p_t][0] for hyp_id, p_t in enumerate(p_ts)])
                    parent_cells = torch.stack([hyp_states[hyp_id][p_t][1] for hyp_id, p_t in enumerate(p_ts)])

                    if args.lstm == 'parent_feed':
                        h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)
                    else:
                        inputs.append(parent_states)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             src_token_mask=None)

            # Variable(batch_size, grammar_size)
            # apply_rule_log_prob = torch.log(F.softmax(self.production_readout(att_t), dim=-1))
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            if args.no_copy:
                primitive_prob = gen_from_vocab_prob
            else:
                # Variable(batch_size, src_sent_len)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

                # Variable(batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # Variable(batch_size, primitive_vocab_size)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

                # if src_unk_pos_list:
                #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_prev_hyp_ids = []

            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                action_types = self.transition_system.get_valid_continuation_types(hyp)

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = apply_rule_log_prob[hyp_id, prod_id].data[0]
                            new_hyp_score = hyp.score + prod_score

                            applyrule_new_hyp_scores.append(new_hyp_score)
                            applyrule_new_hyp_prod_ids.append(prod_id)
                            applyrule_prev_hyp_ids.append(hyp_id)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_log_prob[hyp_id, len(self.grammar)].data[0]
                        new_hyp_score = hyp.score + action_score

                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_new_hyp_prod_ids.append(len(self.grammar))
                        applyrule_prev_hyp_ids.append(hyp_id)
                    else:
                        # GenToken action
                        gentoken_prev_hyp_ids.append(hyp_id)
                        hyp_copy_info = dict()  # of (token_pos, copy_prob)
                        hyp_unk_copy_info = []

                        if args.no_copy is False:
                            for token, token_pos_list in aggregated_primitive_tokens.items():
                                sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0, Variable(T.LongTensor(token_pos_list))).sum()
                                gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                                if token in primitive_vocab:
                                    token_id = primitive_vocab[token]
                                    primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob

                                    hyp_copy_info[token] = (token_pos_list, gated_copy_prob.data[0])
                                else:
                                    hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                              'copy_prob': gated_copy_prob.data[0]})

                        if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                            unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                            token = hyp_unk_copy_info[unk_i]['token']
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']
                            gentoken_new_hyp_unks.append(token)

                            hyp_copy_info[token] = (hyp_unk_copy_info[unk_i]['token_pos_list'], hyp_unk_copy_info[unk_i]['copy_prob'])

            new_hyp_scores = None
            if applyrule_new_hyp_scores:
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores))
            if gentoken_prev_hyp_ids:
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids, :]).view(-1)

                if new_hyp_scores is None: new_hyp_scores = gen_token_new_hyp_scores
                else: new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0), beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                action_info = ActionInfo()
                if new_hyp_pos < len(applyrule_new_hyp_scores):
                    # it's an ApplyRule or Reduce action
                    prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                    prev_hyp = hypotheses[prev_hyp_id]

                    prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                    # ApplyRule action
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    # Reduce action
                    else:
                        action = ReduceAction()
                else:
                    # it's a GenToken action
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1)

                    k = (new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(1)
                    # try:
                    # copy_info = gentoken_copy_infos[k]
                    prev_hyp_id = gentoken_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]
                    # except:
                    #     print('k=%d' % k, file=sys.stderr)
                    #     print('primitive_prob.size(1)=%d' % primitive_prob.size(1), file=sys.stderr)
                    #     print('len copy_info=%d' % len(gentoken_copy_infos), file=sys.stderr)
                    #     print('prev_hyp_id=%s' % ', '.join(str(i) for i in gentoken_prev_hyp_ids), file=sys.stderr)
                    #     print('len applyrule_new_hyp_scores=%d' % len(applyrule_new_hyp_scores), file=sys.stderr)
                    #     print('len gentoken_prev_hyp_ids=%d' % len(gentoken_prev_hyp_ids), file=sys.stderr)
                    #     print('top_new_hyp_pos=%s' % top_new_hyp_pos, file=sys.stderr)
                    #     print('applyrule_new_hyp_scores=%s' % applyrule_new_hyp_scores, file=sys.stderr)
                    #     print('new_hyp_scores=%s' % new_hyp_scores, file=sys.stderr)
                    #     print('top_new_hyp_scores=%s' % top_new_hyp_scores, file=sys.stderr)
                    #
                    #     torch.save((applyrule_new_hyp_scores, primitive_prob), 'data.bin')
                    #
                    #     # exit(-1)
                    #     raise ValueError()

                    if token_id == primitive_vocab.unk_id:
                        if gentoken_new_hyp_unks:
                            token = gentoken_new_hyp_unks[k]
                        else:
                            token = primitive_vocab.id2word[primitive_vocab.unk_id]
                    else:
                        token = primitive_vocab.id2word[token_id]

                    action = GenTokenAction(token)

                    if token in aggregated_primitive_tokens:
                        action_info.copy_from_src = True
                        action_info.src_token_position = aggregated_primitive_tokens[token]

                    if debug:
                        action_info.gen_copy_switch = 'n/a' if args.no_copy else primitive_predictor_prob[prev_hyp_id, :].log().cpu().data.numpy()
                        action_info.in_vocab = token in primitive_vocab
                        action_info.gen_token_prob = gen_from_vocab_prob[prev_hyp_id, token_id].log().cpu().data[0] \
                            if token in primitive_vocab else 'n/a'
                        action_info.copy_token_prob = torch.gather(primitive_copy_prob[prev_hyp_id],
                                                                   0,
                                                                   Variable(T.LongTensor(action_info.src_token_position))).sum().log().cpu().data[0] \
                            if args.no_copy is False and action_info.copy_from_src else 'n/a'

                action_info.action = action
                action_info.t = t
                if t > 0:
                    action_info.parent_t = prev_hyp.frontier_node.created_time
                    action_info.frontier_prod = prev_hyp.frontier_node.production
                    action_info.frontier_field = prev_hyp.frontier_field.field

                if debug:
                    action_info.action_prob = new_hyp_score - prev_hyp.score

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]))
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    @staticmethod
    def load(model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = Parser(saved_args, vocab, transition_system)

        parser.load_state_dict(saved_state)

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser
