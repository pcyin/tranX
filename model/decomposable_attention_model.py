# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import nn_utils
from model.nn_utils import FeedForward


class DecomposableAttentionModel(nn.Module):
    """Decomposable attention model for paraphrase identification"""
    
    def __init__(self, src_vocab, tgt_vocab, embed_size, dropout=0., tie_embed=False, cuda=False):
        super(DecomposableAttentionModel, self).__init__()

        if tie_embed:
            self.src_embed = self.tgt_embed = nn.Embedding(len(src_vocab), embed_size, padding_idx=src_vocab['<pad>'])
        else:
            self.src_embed = nn.Embedding(len(src_vocab), embed_size, padding_idx=src_vocab['<pad>'])
            self.tgt_embed = nn.Embedding(len(tgt_vocab), embed_size, padding_idx=tgt_vocab['<pad>'])

        self.att_linear = nn.Linear(embed_size, embed_size, bias=False)

        self.fuse_func = FeedForward(2 * embed_size, num_layers=2, hidden_dims=embed_size, activations=F.relu, dropout=dropout)
        self.aggregate_ff = FeedForward(2 * embed_size, num_layers=2, hidden_dims=[embed_size, 3], activations=[F.relu, nn_utils.identity], dropout=[dropout, 0.])

    def forward(self, src_sents_var, tgt_sents_var, src_sents_mask, tgt_sents_mask):
        # get input representation
        # (batch_size, src_sent_len, embed_size)
        src_sents_embed = self.encode_sentence(src_sents_var, type='src')
        # (batch_size, tgt_sent_len, embed_size)
        tgt_sents_embed = self.encode_sentence(tgt_sents_var, type='tgt')

        # (batch_size, src_sent_len, embed_size), (batch_size, tgt_sent_len, embed_size)
        src_aligned_phrases, tgt_aligned_phrases = self.get_soft_alignments(src_sents_embed, tgt_sents_embed, src_sents_mask, tgt_sents_mask)

        # (batch_size)
        prob = self.aggregate_and_predict(src_sents_embed, src_aligned_phrases, src_sents_mask,
                                          tgt_sents_embed, tgt_aligned_phrases, tgt_sents_mask)

        return prob

    def encode_sentence(self, src_sents_var, type='src'):
        embed_layer = self.src_embed if type == 'src' else self.tgt_embed
        sents_embed = embed_layer(src_sents_var)

        return sents_embed

    def get_soft_alignments(self, src_sents_embed, tgt_sents_embed, src_sents_mask=None, tgt_sents_mask=None):
        # (batch_size, src_sent_len, tgt_sent_len)
        # (batch_size, tgt_sent_len, src_sent_len)
        src_to_tgt_att_prob, tgt_to_src_att_prob = self.attention(src_sents_embed, tgt_sents_embed, src_sents_mask, tgt_sents_mask)

        # (batch_size, src_sent_len, embed_size)
        betas = torch.bmm(src_to_tgt_att_prob, tgt_sents_embed)

        # (batch_size, tgt_sent_len, embed_size)
        alphas = torch.bmm(tgt_to_src_att_prob, src_sents_embed)

        return betas, alphas

    def attention(self, src_sents_embed, tgt_sents_embed, src_sents_mask=None, tgt_sents_mask=None):
        # (batch_size, src_sent_len, embed_size)
        src_embed_att_linear = self.att_linear(src_sents_embed)

        # (batch_size, src_sent_len, tgt_sent_len)
        att_weights = torch.bmm(src_embed_att_linear, tgt_sents_embed.permute(0, 2, 1))

        if src_sents_mask is None:
            src_to_tgt_att_prob = F.softmax(att_weights, dim=-1)
            # (batch_size, tgt_sent_len, src_sent_len)
            tgt_to_src_att_weights = att_weights.permute(0, 2, 1)
            tgt_to_src_att_prob = F.softmax(tgt_to_src_att_weights, dim=-1)
        else:
            src_sents_mask = src_sents_mask.unsqueeze(1)
            tgt_sents_mask = tgt_sents_mask.unsqueeze(1)

            # (batch_size, src_sent_len, tgt_sent_len)
            src_to_tgt_att_prob = F.softmax(att_weights * tgt_sents_mask, dim=-1)
            src_to_tgt_att_prob = src_to_tgt_att_prob * tgt_sents_mask
            src_to_tgt_att_prob = src_to_tgt_att_prob / (src_to_tgt_att_prob.sum(dim=-1, keepdim=True) + 1e-13)

            # (batch_size, tgt_sent_len, src_sent_len)
            att_weights_t = att_weights.permute(0, 2, 1)
            tgt_to_src_att_prob = F.softmax(att_weights_t * src_sents_mask, dim=-1)
            tgt_to_src_att_prob = tgt_to_src_att_prob * src_sents_mask
            tgt_to_src_att_prob = tgt_to_src_att_prob / (tgt_to_src_att_prob.sum(dim=-1, keepdim=True) + 1e-13)

        return src_to_tgt_att_prob, tgt_to_src_att_prob

    def aggregate_and_predict(self,
                              src_sents_embed, src_aligned_phrases, src_sents_mask,
                              tgt_sents_embed, tgt_aligned_phrases, tgt_sents_mask):
        # (batch_size, [src_sent_len, tgt_sent_len], embed_size)
        v_src = self.fuse_func(torch.cat([src_sents_embed, src_aligned_phrases], dim=-1))
        v_src = v_src * src_sents_mask.unsqueeze(-1)

        v_tgt = self.fuse_func(torch.cat([tgt_sents_embed, tgt_aligned_phrases], dim=-1))
        v_tgt = v_tgt * tgt_sents_mask.unsqueeze(-1)

        v_src = v_src.sum(dim=1)
        v_tgt = v_tgt.sum(dim=1)

        # (batch_size)
        label_logits = self.aggregate_ff(torch.cat([v_src, v_tgt], dim=-1))
        prob = F.log_softmax(label_logits, dim=-1)

        return prob
