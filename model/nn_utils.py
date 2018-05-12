# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

import torch
from torch.autograd import Variable
import numpy as np

from six.moves import xrange


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


def length_array_to_mask_tensor(length_array, cuda=False):
    max_len = length_array[0]
    batch_size = len(length_array)

    mask = np.ones((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        mask[i][:seq_len] = 0

    mask = torch.ByteTensor(mask)
    return mask.cuda() if cuda else mask


def input_transpose(sents, pad_token):
    """
    transform the input List[sequence] of size (batch_size, max_sent_len)
    into a list of size (max_sent_len, batch_size), with proper padding
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in xrange(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in xrange(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in xrange(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def id2word(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab.id2word[w] for w in s] for s in sents]
    else:
        return [vocab.id2word[w] for w in sents]


def to_input_variable(sequences, vocab, cuda=False, training=True, append_boundary_sym=False):
    """
    given a list of sequences,
    return a tensor of shape (max_sent_len, batch_size)
    """
    if append_boundary_sym:
        sequences = [['<s>'] + seq + ['</s>'] for seq in sequences]

    word_ids = word2id(sequences, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=(not training), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def variable_constr(x, v, cuda=False):
    return Variable(torch.cuda.x(v)) if cuda else Variable(torch.x(v))


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in xrange(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]

        yield batch_examples


def isnan(data):
    data = data.cpu().numpy()
    return np.isnan(data).any() or np.isinf(data).any()


def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
       source: https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def uniform_init(lower, upper, params):
    for p in params:
        p.data.uniform_(lower, upper)


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal(p.data)


def identity(x):
    return x
