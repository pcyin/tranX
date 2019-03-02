# coding=utf-8
from __future__ import print_function

from itertools import chain

from asdl.hypothesis import Hypothesis
from asdl.lang.sql.sql_transition_system import WikiSqlSelectColumnAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, GenTokenAction
from components.action_info import ActionInfo
from components.decode_hypothesis import DecodeHypothesis
from model import nn_utils
from model.parser import Parser

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model.pointer_net import PointerNet
from model.wikisql.dataset import WikiSqlBatch
from common.registerable import Registrable


@Registrable.register('wikisql_parser')
class WikiSqlParser(Parser):
    def __init__(self, args, vocab, transition_system):
        super(WikiSqlParser, self).__init__(args, vocab, transition_system)

        self.table_header_lstm = nn.LSTM(args.embed_size, int(args.hidden_size / 2), bidirectional=True, batch_first=True)
        self.column_pointer_net = PointerNet(args.hidden_size, args.hidden_size, attention_type=args.column_att)

        self.column_rnn_input = nn.Linear(args.hidden_size, args.embed_size, bias=False)

    def encode_table_header(self, tables):
        # input, ids of table word: (batch_size, max_column_num)
        # encode_output: (max_head_word_num, batch_size, max_column_num, hidden_size)

        # (batch_size, max_column_num, max_head_word_num)
        # table_head_mask: (batch_size, max_column_num)
        # table_col_lens: (batch_size, max_column_num)
        table_head_wids, table_col_lens = WikiSqlBatch.get_table_header_input_tensor(tables,
                                                                                     self.vocab.source,
                                                                                     cuda=self.args.cuda)

        # hack: pack_padded_sequence requires seq length to be greater than 1
        for tbl in table_col_lens:
            for i in range(len(tbl)):
                if tbl[i] == 0: tbl[i] = 1

        table_header_mask = WikiSqlBatch.get_table_header_mask(tables, cuda=self.args.cuda)

        # (batch_size, max_column_num, max_head_word_num, word_embed_size)
        table_head_word_embeds = self.src_embed(table_head_wids.view(-1)).view(list(table_head_wids.size()) + [self.src_embed.embedding_dim])

        batch_size = table_head_word_embeds.size(0)
        max_col_num = table_head_word_embeds.size(1)
        max_col_word_num = table_head_word_embeds.size(2)

        # (batch_size * max_column_num, max_head_word_num, word_embed_size)
        table_head_word_embeds_flatten = table_head_word_embeds.view(batch_size * max_col_num,
                                                                     max_col_word_num, -1)
        table_col_lens_flatten = list(chain.from_iterable(table_col_lens))
        sorted_col_ids = sorted(list(range(len(table_col_lens_flatten))), key=lambda x: -table_col_lens_flatten[x])
        sorted_table_col_lens_flatten = [table_col_lens_flatten[i] for i in sorted_col_ids]

        col_old_pos_map = [-1] * len(sorted_col_ids)
        for new_pos, old_pos in enumerate(sorted_col_ids):
            col_old_pos_map[old_pos] = new_pos

        # (batch_size * max_column_num, max_head_word_num, word_embed_size)
        sorted_table_head_word_embeds = table_head_word_embeds_flatten[sorted_col_ids, :, :]

        packed_table_head_word_embeds = pack_padded_sequence(sorted_table_head_word_embeds, sorted_table_col_lens_flatten, batch_first=True)

        # column_word_encodings: (batch_size * max_column_num, max_head_word_num, hidden_size)
        column_word_encodings, (table_header_encoding, table_head_last_cell) = self.table_header_lstm(packed_table_head_word_embeds)
        column_word_encodings, _ = pad_packed_sequence(column_word_encodings, batch_first=True)

        # (batch_size * max_column_num, max_head_word_num, hidden_size)
        column_word_encodings = column_word_encodings[col_old_pos_map]
        # (batch_size, max_column_num, max_head_word_num, hidden_size)
        column_word_encodings = column_word_encodings.view(batch_size, max_col_num, max_col_word_num, -1)

        # (batch_size, hidden_size * 2)
        table_header_encoding = torch.cat([table_header_encoding[0], table_header_encoding[1]], -1)
        # table_head_last_cell = torch.cat([table_head_last_cell[0], table_head_last_cell[1]], -1)

        # same
        table_header_encoding = table_header_encoding[col_old_pos_map]
        # (batch_size, max_column_num, hidden_size)
        table_header_encoding = table_header_encoding.view(batch_size, max_col_num, -1)

        return column_word_encodings, table_header_encoding, table_header_mask

    def score(self, examples, return_encode_state=False):
        """
        input: a batch of examples
        output: score for each training example: Variable(batch_size)
        """
        args = self.args
        batch = WikiSqlBatch(examples, self.grammar, self.vocab, cuda=self.args.cuda)

        src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        column_word_encodings, table_header_encoding, table_header_mask = self.encode_table_header([e.table for e in examples])

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size)
        utterance_encodings_att_linear = self.att_src_linear(src_encodings)

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        history_states = []
        att_vecs = []

        action_probs = [[] for example in examples]

        for t in range(batch.max_action_num):
            # x: [prev_action, parent_production_embed, parent_field_embed, parent_field_type_embed, parent_action_state]
            if t == 0:
                x = Variable(self.new_tensor(len(batch), self.decoder_lstm.input_size).zero_(), requires_grad=False)
                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.hidden_size * (not args.no_input_feed)
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[:, offset: offset + args.type_embed_size] = self.type_embed(Variable(self.new_long_tensor(
                        [self.grammar.type2id[self.grammar.root_type] for e in batch.examples])))
            else:
                a_tm1_embeds = []
                for e_id, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_info_tm1 = example.tgt_actions[t - 1]
                        action_tm1 = action_info_tm1.action
                        if isinstance(action_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        elif isinstance(action_tm1, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        elif isinstance(action_tm1, WikiSqlSelectColumnAction):
                            a_tm1_embed = self.column_rnn_input(table_header_encoding[e_id, action_tm1.column_id])
                        elif isinstance(action_tm1, GenTokenAction):
                            a_tm1_embed = self.src_embed.weight[self.vocab.source[action_tm1.token]]
                        else:
                            raise ValueError('unknown action %s' % action_tm1)
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

            (h_t, cell_t), att_t = self.step(x, h_tm1, src_encodings,
                                             utterance_encodings_att_linear,
                                             src_token_mask=batch.src_token_mask)

            # ApplyRule action probability
            # (batch_size, grammar_size)
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            # column attention
            # (batch_size, max_head_num)
            column_attention_weights = self.column_pointer_net(table_header_encoding, table_header_mask, att_t.unsqueeze(0)).squeeze(0)

            if any(type(e.tgt_actions[t].action) is GenTokenAction for e in examples if t < len(e.tgt_actions)):
                # (batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # primitive copy prob
                # (batch_size, src_token_num)
                primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask,
                                                           att_t.unsqueeze(0)).squeeze(0)

                # (batch_size, primitive_vocab_size)
                primitive_gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_info_t = example.tgt_actions[t]
                    action_t = action_info_t.action

                    if isinstance(action_t, ApplyRuleAction):
                        act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    elif isinstance(action_t, ReduceAction):
                        act_prob_t_i = apply_rule_prob[e_id, len(self.grammar)]
                    elif isinstance(action_t, WikiSqlSelectColumnAction):
                        # select a column using column attention
                        act_prob_t_i = column_attention_weights[e_id, action_t.column_id]
                    elif isinstance(action_t, GenTokenAction):
                        # value words can only be copied from the source question!
                        # but here we still implement in a rather general fashion

                        token_id = self.vocab.primitive[action_t.token]
                        if action_info_t.copy_from_src:
                            if action_t.token in self.vocab.primitive:
                                act_prob_t_i = primitive_predictor_prob[e_id, 0] * primitive_gen_from_vocab_prob[e_id, token_id] + \
                                               primitive_predictor_prob[e_id, 1] * primitive_copy_prob[e_id, action_info_t.src_token_position]
                            else:
                                act_prob_t_i = primitive_predictor_prob[e_id, 1] * primitive_copy_prob[e_id, action_info_t.src_token_position]
                        else:
                            act_prob_t_i = primitive_predictor_prob[e_id, 0] * primitive_gen_from_vocab_prob[e_id, token_id]
                    else:
                        raise ValueError('unknown action %s' % action_t)

                    action_probs[e_id].append(act_prob_t_i)

            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        # sum all the action probabilities
        action_prob_var = torch.cat([torch.cat(action_probs_i).log().sum() for action_probs_i in action_probs])

        return [action_prob_var]   # TODO: supervised attention not implemented yet!

    def parse(self, question, context, beam_size=5):
        table = context
        args = self.args
        src_sent_var = nn_utils.to_input_variable([question], self.vocab.source,
                                                  cuda=self.args.cuda, training=False)

        utterance_encodings, (last_state, last_cell) = self.encode(src_sent_var, [len(question)])
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        column_word_encodings, table_header_encoding, table_header_mask = self.encode_table_header([table])

        h_tm1 = dec_init_vec
        # (batch_size, query_len, hidden_size)
        utterance_encodings_att_linear = self.att_src_linear(utterance_encodings)

        zero_action_embed = Variable(self.new_tensor(self.args.action_embed_size).zero_())

        t = 0
        hypotheses = [DecodeHypothesis()]
        hyp_states = [[]]
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < self.args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = utterance_encodings.expand(hyp_num, utterance_encodings.size(1), utterance_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = utterance_encodings_att_linear.expand(hyp_num,
                                                                                 utterance_encodings_att_linear.size(1),
                                                                                 utterance_encodings_att_linear.size(2))

            # x: [prev_action, parent_production_embed, parent_field_embed, parent_field_type_embed, parent_action_state]
            if t == 0:
                x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).zero_(), volatile=True)

                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.hidden_size * (not args.no_input_feed)
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[0, offset: offset + args.type_embed_size] = \
                        self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
            else:
                a_tm1_embeds = []
                for e_id, hyp in enumerate(hypotheses):
                    action_tm1 = hyp.actions[-1]
                    if action_tm1:
                        if isinstance(action_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        elif isinstance(action_tm1, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        elif isinstance(action_tm1, WikiSqlSelectColumnAction):
                            a_tm1_embed = self.column_rnn_input(table_header_encoding[0, action_tm1.column_id])
                        elif isinstance(action_tm1, GenTokenAction):
                            a_tm1_embed = self.src_embed.weight[self.vocab.source[action_tm1.token]]
                        else:
                            raise ValueError('unknown action %s' % action_tm1)
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

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

            # ApplyRule action probability
            # (batch_size, grammar_size)
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            # column attention
            # (batch_size, max_head_num)
            column_attention_weights = self.column_pointer_net(table_header_encoding, table_header_mask,
                                                               att_t.unsqueeze(0)).squeeze(0)
            column_selection_log_prob = torch.log(column_attention_weights)

            # (batch_size, 2)
            primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

            # primitive copy prob
            # (batch_size, src_token_num)
            primitive_copy_prob = self.src_pointer_net(utterance_encodings, None,
                                                       att_t.unsqueeze(0)).squeeze(0)

            # (batch_size, primitive_vocab_size)
            primitive_gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            new_hyp_meta = []

            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                action_types = self.transition_system.get_valid_continuation_types(hyp)

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = apply_rule_log_prob[hyp_id, prod_id]
                            new_hyp_score = hyp.score + prod_score

                            meta_entry = {'action_type': 'apply_rule', 'prod_id': prod_id,
                                          'score': prod_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_log_prob[hyp_id, len(self.grammar)]
                        new_hyp_score = hyp.score + action_score

                        meta_entry = {'action_type': 'apply_rule', 'prod_id': len(self.grammar),
                                      'score': action_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                    elif action_type == WikiSqlSelectColumnAction:
                        for col_id, column in enumerate(table.header):
                            col_sel_score = column_selection_log_prob[hyp_id, col_id]
                            new_hyp_score = hyp.score + col_sel_score

                            meta_entry = {'action_type': 'sel_col', 'col_id': col_id,
                                          'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)
                    elif action_type == GenTokenAction:
                        # remember that we can only copy stuff from the input!
                        # we only copy tokens sequentially!!
                        prev_action = hyp.action_infos[-1].action

                        valid_token_pos_list = []
                        if type(prev_action) is GenTokenAction and \
                                not prev_action.is_stop_signal():
                            token_pos = hyp.action_infos[-1].src_token_position + 1
                            if token_pos < len(question):
                                valid_token_pos_list = [token_pos]
                        else:
                            valid_token_pos_list = list(range(len(question)))

                        col_id = hyp.frontier_node['col_idx'].value
                        if table.header[col_id].type == 'real':
                            valid_token_pos_list = [i for i in valid_token_pos_list
                                                    if any(c.isdigit() for c in question[i]) or
                                                    hyp._value_buffer and question[i] in (',', '.', '-', '%')]

                        p_copies = primitive_predictor_prob[hyp_id, 1] * primitive_copy_prob[hyp_id]
                        for token_pos in valid_token_pos_list:
                            token = question[token_pos]
                            p_copy = p_copies[token_pos]
                            score_copy = torch.log(p_copy)

                            meta_entry = {'action_type': 'gen_token',
                                          'token': token, 'token_pos': token_pos,
                                          'score': score_copy, 'new_hyp_score': score_copy + hyp.score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)

                        # add generation probability for </primitive>
                        if hyp._value_buffer:
                            eos_prob = primitive_predictor_prob[hyp_id, 0] * \
                                       primitive_gen_from_vocab_prob[hyp_id, self.vocab.primitive['</primitive>']]
                            eos_score = torch.log(eos_prob)

                            meta_entry = {'action_type': 'gen_token',
                                          'token': '</primitive>',
                                          'score': eos_score, 'new_hyp_score': eos_score + hyp.score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)

            if not new_hyp_meta: break

            new_hyp_scores = torch.cat([x['new_hyp_score'] for x in new_hyp_meta])
            top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores,
                                                      k=min(new_hyp_scores.size(0),
                                                            beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = hypotheses[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if action_type_str == 'apply_rule':
                    # ApplyRule action
                    prod_id = hyp_meta_entry['prod_id']
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    # Reduce action
                    else:
                        action = ReduceAction()
                elif action_type_str == 'sel_col':
                    action = WikiSqlSelectColumnAction(hyp_meta_entry['col_id'])
                else:
                    action = GenTokenAction(hyp_meta_entry['token'])
                    if 'token_pos' in hyp_meta_entry:
                        action_info.copy_from_src = True
                        action_info.src_token_position = hyp_meta_entry['token_pos']

                action_info.action = action
                action_info.t = t

                if t > 0:
                    action_info.parent_t = prev_hyp.frontier_node.created_time
                    action_info.frontier_prod = prev_hyp.frontier_node.production
                    action_info.frontier_field = prev_hyp.frontier_field.field

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
                t += 1
            else: break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses
