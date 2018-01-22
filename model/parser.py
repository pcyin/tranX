# coding=utf-8
import os

import path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from asdl.hypothesis import Hypothesis, GenTokenAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, Action
from components.dataset import Batch
from model import nn_utils
from model.pointer_net import PointerNet


class Parser(nn.Module):
    def __init__(self, args, vocab, transition_system):
        super(Parser, self).__init__()

        self.args = args
        self.vocab = vocab

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        # Embedding layers
        self.src_embed = nn.Embedding(len(vocab.source), args.embed_size)
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)
        self.primitive_embed = nn.Embedding(len(vocab.primitive), args.action_embed_size)
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        nn.init.xavier_normal(self.src_embed.weight.data)
        nn.init.xavier_normal(self.production_embed.weight.data)
        nn.init.xavier_normal(self.primitive_embed.weight.data)
        nn.init.xavier_normal(self.field_embed.weight.data)
        nn.init.xavier_normal(self.type_embed.weight.data)

        # LSTMs
        if args.lstm == 'lstm':
            self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size / 2, bidirectional=True)
            self.decoder_lstm = nn.LSTMCell(args.action_embed_size +  # previous action
                                         args.action_embed_size + args.field_embed_size + args.type_embed_size +  # frontier info
                                         args.hidden_size,  # parent hidden state
                                         args.hidden_size)
        else:
            from lstm import LSTM, LSTMCell
            self.encoder_lstm = LSTM(args.embed_size, args.hidden_size / 2, bidirectional=True, dropout=args.dropout)
            self.decoder_lstm = LSTMCell(args.action_embed_size +   # previous action
                                         args.action_embed_size + args.field_embed_size + args.type_embed_size +  # frontier info
                                         args.hidden_size,   # parent hidden state
                                         args.hidden_size,
                                         dropout=args.dropout)

        # pointer net
        self.src_pointer_net = PointerNet(args)

        self.primitive_predictor = nn.Linear(args.hidden_size, 2)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.hidden_size, bias=False)

        # embedding layers
        self.production_readout = nn.Linear(args.hidden_size, len(transition_system.grammar) + 1)
        self.tgt_token_readout = nn.Linear(args.hidden_size, len(vocab.primitive))

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def encode(self, src_sents_var, src_sents_len):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
            last_state, last_cell: Variable(batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
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
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = F.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

    def score(self, examples):
        """
        input: a batch of examples
        output: score for each training example: Variable(batch_size)
        """
        batch = Batch(examples, self.grammar, self.vocab, self.args.cuda)
        src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        # Variable(tgt_action_len, batch_size, hidden_size)
        query_vectors = self.decode(batch, src_encodings, dec_init_vec)

        # ApplyRule action probability
        # (tgt_action_len, batch_size, grammar_size)
        apply_rule_prob = F.log_softmax(self.production_readout(query_vectors), dim=-1)

        # pointer network scores over source tokens
        # Variable(tgt_action_len, batch_size, src_sent_len)
        primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)

        # Variable(tgt_action_len, batch_size, primitive_vocab_size)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # Variable(tgt_action_len, batch_size, 2)
        primitive_predictor_prob = F.softmax(self.primitive_predictor(query_vectors), dim=-1)

        # (tgt_action_len, batch_size)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2, index=batch.apply_rule_idx_matrix.unsqueeze(2)).squeeze(2)

        # (tgt_action_len, bathc_size)
        tgt_primitive_copy_prob = torch.gather(primitive_copy_prob, dim=2, index=batch.primitive_copy_pos_matrix.unsqueeze(2)).squeeze(2)

        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(2)).squeeze(2)

        # (tgt_action_len, batch_size)
        action_prob = tgt_apply_rule_prob * batch.apply_rule_mask + \
                      torch.log(primitive_predictor_prob[:, :, 0] * tgt_primitive_gen_from_vocab_prob) * batch.gen_token_mask + \
                      torch.log(primitive_predictor_prob[:, :, 1] * tgt_primitive_copy_prob) * batch.primitive_copy_mask

        loss = torch.sum(action_prob, dim=0)

        return loss

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_token_mask=None):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t

    def decode(self, batch, src_encodings, dec_init_vec):
        batch_size = len(batch)
        args = self.args

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        att_vecs = []
        history_states = []

        if args.lstm == 'lstm_with_dropout':
            self.decoder_lstm.set_dropout_masks(batch_size)

        for t in xrange(batch.max_action_num):
            # x: [prev_action, parent_production_embed, parent_field_embed, parent_field_type_embed, parent_action_state]
            if t == 0:
                x = Variable(self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_(), requires_grad=False)
                offset = args.action_embed_size * 2 + args.field_embed_size
                x[:, offset: offset + args.type_embed_size] = self.type_embed(Variable(self.new_long_tensor(
                    [self.grammar.type2id[self.grammar.root_type] for e in batch.examples])))
            else:
                actions_tm1 = [e.tgt_actions[t - 1] if t < len(e.tgt_actions) else None for e in batch.examples]
                actions_t = [e.tgt_actions[t] if t < len(e.tgt_actions) else None for e in batch.examples]

                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1.action, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.action.production]]
                        elif isinstance(a_tm1.action, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.action.token]]

                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                x = torch.cat([a_tm1_embeds,
                               self.production_embed(batch.get_frontier_prod_idx(t)),
                               self.field_embed(batch.get_frontier_field_idx(t)),
                               self.type_embed(batch.get_frontier_field_type_idx(t))], dim=-1)

                # append history states
                parent_states = torch.stack([history_states[p_t][batch_id]
                                             for batch_id, p_t in
                                             enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])

                x = torch.cat([x, parent_states], dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, src_encodings,
                                             src_encodings_att_linear,
                                             src_token_mask=batch.src_token_mask)

            history_states.append(h_t)
            att_vecs.append(att_t)
            h_tm1 = (h_t, cell_t)

        att_vecs = torch.stack(att_vecs)
        return att_vecs

    def parse(self, src_sent, beam_size=5):
        args = self.args
        primitive_vocab = self.vocab.primitive

        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, cuda=args.cuda, training=False)

        # Variable(1, src_sent_len, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encode(src_sent_var, [len(src_sent)])
        # (1, src_sent_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = self.init_decoder_state(last_state, last_cell)
        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        hyp_scores = Variable(self.new_tensor([0.]), volatile=True)

        src_token_vocab_ids = [primitive_vocab[token] for token in src_sent]
        src_unk_pos_list = [pos for pos, token_id in enumerate(src_token_vocab_ids) if token_id == primitive_vocab.unk_id]
        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_vocab_ids):
            if tid in token_set:
                src_token_vocab_ids[i] = -1
            else: token_set.add(tid)

        t = 0
        hypotheses = [Hypothesis()]
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
                offset = args.action_embed_size * 2 + args.field_embed_size
                x[0, offset: offset + args.type_embed_size] = self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
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

                # frontier production
                frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                frontier_prod_embeds = self.production_embed(Variable(self.new_long_tensor(
                    [self.grammar.prod2id[prod] for prod in frontier_prods])))

                # frontier field
                frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                frontier_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                    self.grammar.field2id[field] for field in frontier_fields])))

                # frontier field type
                frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                frontier_field_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                    self.grammar.type2id[type] for type in frontier_field_types])))

                # parent states
                p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                hist_states = torch.stack([hyp_states[hyp_id][p_t] for hyp_id, p_t in enumerate(p_ts)])

                x = torch.cat([a_tm1_embeds,
                               frontier_prod_embeds,
                               frontier_field_embeds,
                               frontier_field_type_embeds,
                               hist_states], dim=-1)

            if args.lstm == 'lstm_with_dropout':
                self.decoder_lstm.set_dropout_masks(hyp_num)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             src_token_mask=None)

            # Variable(batch_size, grammar_size)
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            # Variable(batch_size, src_sent_len)
            primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            # Variable(batch_size, 2)
            primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

            # Variable(batch_size, primitive_vocab_size)
            primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob
            primitive_prob[:, primitive_vocab.unk_id] = 0.

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
                        # first, we compute copy probabilities for tokens in the source sentence
                        for token_pos, token_vocab_id in enumerate(src_token_vocab_ids):
                            if token_vocab_id != -1:
                                primitive_prob[hyp_id, token_vocab_id] = primitive_prob[hyp_id, token_vocab_id] + \
                                                                         primitive_predictor_prob[hyp_id, 1] * primitive_copy_prob[hyp_id, token_pos]

                        # second, add the probability of copying the most probable unk word
                        if src_unk_pos_list:
                            unk_pos = primitive_copy_prob[hyp_id][src_unk_pos_list].data.cpu().numpy().argmax()
                            unk_pos = src_unk_pos_list[unk_pos]
                            token = src_sent[unk_pos]
                            gentoken_new_hyp_unks.append(token)

                            unk_copy_score = primitive_predictor_prob[hyp_id, 1] * primitive_copy_prob[hyp_id, unk_pos]
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = unk_copy_score

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

                    new_hyp = prev_hyp.clone_and_apply_action(action)
                    new_hyp.score = new_hyp_score
                else:
                    # it's a GenToken action
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1)

                    k = (new_hyp_pos - len(applyrule_new_hyp_scores)) / primitive_prob.size(1)
                    prev_hyp_id = gentoken_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]

                    if token_id == primitive_vocab.unk_id:
                        token = gentoken_new_hyp_unks[k]
                    else:
                        token = primitive_vocab.id2word[token_id]

                    action = GenTokenAction(token)
                    new_hyp = prev_hyp.clone_and_apply_action(action)
                    new_hyp.score = new_hyp_score

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [h_t[i]] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
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
