# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, hidden_dim):
        super(PointerNet, self).__init__()

        self.src_encoding_linear = nn.Linear(src_encoding_size, hidden_dim)
        self.query_vec_linear = nn.Linear(query_vec_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(src_sent_len, batch_size)
        :param query_vec: Variable(tgt_action_num, batch_size, hidden_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """

        # (tgt_action_num, batch_size, src_sent_len, ptrnet_hidden_dim)
        h1 = torch.tanh(self.src_encoding_linear(src_encodings).unsqueeze(0) + self.query_vec_linear(query_vec).unsqueeze(2))
        # (tgt_action_num, batch_size, src_sent_len)
        h2 = self.layer2(h1).squeeze(3)
        if src_token_mask is not None:
            # (tgt_action_num, batch_size, src_sent_len)
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(h2)
            h2.data.masked_fill_(src_token_mask, -float('inf'))

        ptr_weights = F.softmax(h2, dim=-1)

        return ptr_weights
