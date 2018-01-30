# coding=utf-8
# coding=utf-8
from __future__ import print_function

import argparse
import cPickle as pickle
import traceback
from itertools import ifilter

import numpy as np
import time
import math

import sys
import torch
from torch.autograd import Variable

import evaluation
from asdl.asdl import ASDLGrammar
from asdl.lang.py.py_transition_system import PythonTransitionSystem
from components.dataset import Dataset
from model import nn_utils
from model.neural_lm import LSTMLanguageModel

from model.parser import Parser
from model.prior import UniformPrior, LSTMPrior
from model.reconstruction_model import Reconstructor
from model.struct_vae import StructVAE


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')

    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')

    parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--train_file', type=str, help='path to the training target file')
    parser.add_argument('--dev_file', type=str, help='path to the dev source file')

    parser.add_argument('--log_every', default=10, type=int, help='every n iterations to log training statistics')

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


def train(args):
    train_set = Dataset.from_bin_file(args.train_file)
    dev_set = Dataset.from_bin_file(args.dev_file)
    vocab = pickle.load(open(args.vocab))

    model = LSTMLanguageModel(vocab.source, args.embed_size, args.hidden_size, dropout=args.dropout)
    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def evaluate_ppl():
        model.eval()
        cum_loss = 0.
        cum_tgt_words = 0.
        for batch in dev_set.batch_iter(args.batch_size):
            src_sents_var = nn_utils.to_input_variable([e.src_sent for e in batch], vocab.source, cuda=args.cuda,
                                                       append_boundary_sym=True)
            loss = model(src_sents_var).sum()
            cum_loss += loss.data[0]
            cum_tgt_words += sum(len(e.src_sent) + 1 for e in batch)  # add ending </s>

        ppl = np.exp(cum_loss / cum_tgt_words)
        model.train()
        return ppl

    print('begin training decoder, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab size: %d' % len(vocab.source), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= 100]
            src_sents = [e.src_sent for e in batch_examples]
            src_sents_var = nn_utils.to_input_variable(src_sents, vocab.source, cuda=args.cuda, append_boundary_sym=True)

            train_iter += 1
            optimizer.zero_grad()

            loss = model(src_sents_var)
            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                print('[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr)

                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)
        # model_file = args.save_to + '.iter%d.bin' % train_iter
        # print('save model to [%s]' % model_file, file=sys.stderr)
        # model.save(model_file)

        # perform validation
        print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
        eval_start = time.time()
        # evaluate ppl
        ppl = evaluate_ppl()
        print('[Epoch %d] ppl=%.5f took %ds' % (epoch, ppl, time.time() - eval_start), file=sys.stderr)
        dev_acc = -ppl
        is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
        history_dev_scores.append(dev_acc)

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save currently the best model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

            if patience == args.patience:
                num_trial += 1
                print('hit #%d trial' % num_trial, file=sys.stderr)
                if num_trial == args.max_num_trial:
                    print('early stop!', file=sys.stderr)
                    exit(0)

                # decay lr, and restore from previously best checkpoint
                lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                # load model
                params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                if args.cuda: model = model.cuda()

                # load optimizers
                if args.reset_optimizer:
                    print('reset optimizer', file=sys.stderr)
                    optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
                else:
                    print('restore parameters of the optimizers', file=sys.stderr)
                    optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

                # set new lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # reset patience
                patience = 0

if __name__ == '__main__':
    args = init_config()
    train(args)
