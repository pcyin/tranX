# coding=utf-8
from __future__ import print_function

import ast
import fnmatch
import traceback

import astor
import time

import sys
import os
from asdl.lang.py.dataset import Django
from asdl.lang.py.py_utils import tokenize_code
from components.vocab import *
from model import nn_utils
from model.neural_lm import LSTMLanguageModel

import numpy as np

from model.prior import LSTMPrior


def load_code_dir(code_dir):
    all_code = []
    for root, dirnames, filenames in os.walk(code_dir):
        for filename in fnmatch.filter(filenames, '*.py'):
            file_path = os.path.join(root, filename)
            print('parsing %s' % file_path, file=sys.stderr)
            code_data = load_code_data(file_path)
            all_code.extend(code_data)

    return all_code


def load_code_data(file_path):
    # read in lines of Python code to train a prior
    processed_code = []
    for line in open(file_path):
        raw_code = line.strip()
        # perform canonicalization same as how we pre-process the dataset
        if raw_code:
            code = Django.canonicalize_code(raw_code)
            try:
                # use the astor-style code
                py_ast = ast.parse(code).body[0]
                code = astor.to_source(py_ast).strip()
                code_tokens = tokenize_code(code, mode='canonicalize')
                if len(code_tokens) < 50:
                    processed_code.append({'code': code, 'tokens': code_tokens})
            except:
                print("Exception in reading line: %s" % raw_code, file=sys.stdout)
                print('-' * 60, file=sys.stdout)
                traceback.print_exc(file=sys.stdout)
                print('-' * 60, file=sys.stdout)

    return processed_code


def train_lstm_lm(args):
    all_data = load_code_dir(args.code_dir)
    np.random.shuffle(all_data)
    train_data = all_data[:-1000]
    dev_data = all_data[-1000:]
    print('train data size: %d, dev data size: %d' % (len(train_data), len(dev_data)), file=sys.stderr)

    vocab = VocabEntry.from_corpus([e['tokens'] for e in train_data], size=args.vocab_size, freq_cutoff=args.freq_cutoff)
    print('vocab size: %d' % len(vocab), file=sys.stderr)

    model = LSTMPrior(args, vocab)
    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    def evaluate_ppl():
        model.eval()
        cum_loss = 0.
        cum_tgt_words = 0.
        for examples in nn_utils.batch_iter(dev_data, args.batch_size):
            batch_tokens = [e['tokens'] for e in examples]
            batch = nn_utils.to_input_variable(batch_tokens, vocab, cuda=args.cuda, append_boundary_sym=True)
            loss = model.forward(batch).sum()
            cum_loss += loss.data[0]
            cum_tgt_words += sum(len(tokens) + 1 for tokens in batch_tokens)  # add ending </s>

        ppl = np.exp(cum_loss / cum_tgt_words)
        model.train()
        return ppl

    print('begin training decoder, %d training examples, %d dev examples' % (len(train_data), len(dev_data)),
          file=sys.stderr)

    epoch = num_trial = train_iter = patience = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    while True:
        epoch += 1
        epoch_begin = time.time()

        for examples in nn_utils.batch_iter(train_data, batch_size=args.batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()

            batch_tokens = [e['tokens'] for e in examples]
            batch = nn_utils.to_input_variable(batch_tokens, vocab, cuda=args.cuda, append_boundary_sym=True)
            loss = model.forward(batch)
            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(examples)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--embed_size', default=128, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0.4, type=float, help='dropout rate')
    parser.add_argument('--vocab_size', default=10000, type=int, help='vocab size')
    parser.add_argument('--freq_cutoff', default=5, type=int, help='vocab size')

    parser.add_argument('--code_dir', type=str)
    parser.add_argument('--train_file', type=str, help='path to the training target file')
    parser.add_argument('--dev_file', type=str, help='path to the dev source file')

    parser.add_argument('--log_every', default=10, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--max_num_trial', default=10, type=int)
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay learning rate if the validation performance drops')
    parser.add_argument('--reset_optimizer', action='store_true', default=False)

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 / 7)

    train_lstm_lm(args)
