# coding=utf-8
from __future__ import print_function

import argparse
import cPickle as pickle
import numpy as np
import time
import math

import sys
import torch

from asdl.asdl import ASDLGrammar
from asdl.lang.py.py_transition_system import PythonTransitionSystem
from components.dataset import Dataset

from model.parser import Parser


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    parser.add_argument('--mode', choices=['train', 'train_semi', 'test', 'debug_ls'], default='train', help='run mode')

    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--sample_size', default=5, type=int, help='sample size')
    parser.add_argument('--embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--action_embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--field_embed_size', default=64, type=int, help='size of word embeddings')
    parser.add_argument('--type_embed_size', default=64, type=int, help='size of word embeddings')
    parser.add_argument('--ptrnet_hidden_dim', default=32, type=int)
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')
    parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='word dropout on decoder')
    parser.add_argument('--kl_anneal', default=False, action='store_true')
    parser.add_argument('--alpha', default=0.1, type=float)

    parser.add_argument('--asdl_file', type=str)
    parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--train_src', type=str, help='path to the training source file')
    parser.add_argument('--unlabeled_src', type=str, help='path to the training source file')
    parser.add_argument('--unlabeled_tgt', type=str, default=None, help='path to the target file')
    parser.add_argument('--train_file', type=str, help='path to the training target file')
    parser.add_argument('--dev_file', type=str, help='path to the dev source file')
    parser.add_argument('--test_file', type=str, help='path to the test target file')
    parser.add_argument('--prior_lm_path', type=str, help='path to the prior LM')

    # semi-supervised learning arguments
    parser.add_argument('--begin_semisup_after_dev_acc', type=float, default=0., help='begin semi-supervised learning after'
                                                                                    'we have reached certain dev performance')

    parser.add_argument('--decode_max_time_step', default=80, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')
    parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')

    parser.add_argument('--valid_metric', default='sp_acc', choices=['nlg_bleu', 'sp_acc'],
                        help='metric used for validation')
    parser.add_argument('--log_every', default=10, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--save_decode_to', default=None, type=str, help='save decoding results to file')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--max_num_trial', default=10, type=int)
    parser.add_argument('--uniform_init', default=None, type=float,
                        help='if specified, use uniform initialization for all parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay learning rate if the validation performance drops')
    parser.add_argument('--lr_decay_after_epoch', default=5, type=int)
    parser.add_argument('--reset_optimizer', action='store_true', default=False)

    parser.add_argument('--train_opt', default="reinforce", type=str, choices=['reinforce', 'st_gumbel'])

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 / 7)

    return args

if __name__ == '__main__':
    args = init_config()

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = PythonTransitionSystem(grammar)
    train_set = Dataset.from_bin_file(args.train_file)
    vocab = pickle.load(open(args.vocab))

    parser = Parser(args, vocab, transition_system)
    parser.train()
    if args.cuda: parser.cuda()
    optimizer = torch.optim.Adam(parser.parameters(), lr=args.lr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= 100]
            train_iter += 1
            optimizer.zero_grad()

            loss = -parser.score(batch_examples)
            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(parser.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                print('[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr)

                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
