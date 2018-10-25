# coding=utf-8
from __future__ import print_function

import argparse
from itertools import chain

import six.moves.cPickle as pickle
from six.moves import xrange as range
from six.moves import input
import traceback

import numpy as np
import time
import os
import sys

import torch
from torch.autograd import Variable

import evaluation
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from components.dataset import Dataset, Example
from components.standalone_parser import StandaloneParser
from model import nn_utils, utils
from model.neural_lm import LSTMLanguageModel

from model.parser import Parser
from model.prior import UniformPrior, LSTMPrior
from model.reconstruction_model import Reconstructor
from model.struct_vae import StructVAE, StructVAE_LMBaseline, StructVAE_SrcLmAndLinearBaseline
from model.utils import GloveHelper, get_parser_class


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--lang', choices=['python', 'lambda_dcs', 'wikisql', 'prolog'], default='python')
    arg_parser.add_argument('--asdl_file', type=str, help='Path to ASDL grammar specification')
    arg_parser.add_argument('--mode', choices=['train', 'self_train', 'train_reconstructor',
                                               'test', 'rerank', 'interactive'], default='train', help='Run mode')

    #### Model configuration ####
    arg_parser.add_argument('--lstm', choices=['lstm'], default='lstm', help='Type of LSTM used, currently only standard LSTM cell is supported')

    # Embedding sizes
    arg_parser.add_argument('--embed_size', default=128, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='Embedding size of ASDL fields')
    arg_parser.add_argument('--type_embed_size', default=64, type=int, help='Embeddings ASDL types')

    # Hidden sizes
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--ptrnet_hidden_dim', default=32, type=int, help='Hidden dimension used in pointer network')
    arg_parser.add_argument('--att_vec_size', default=256, type=int, help='size of attentional vector')

    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true',
                            help='Do not use additional linear layer to transform the attentional vector for computing action probabilities')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'],
                            help='Type of activation if using additional linear layer')
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true',
                            help='Use different linear mapping ')

    # supervised attention
    arg_parser.add_argument('--sup_attention', default=False, action='store_true', help='Use supervised attention')

    # parent information switch for decoder LSTM
    arg_parser.add_argument('--no_parent_production_embed', default=False, action='store_true',
                            help='Do not use embedding of parent ASDL production to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_embed', default=False, action='store_true',
                            help='Do not use embedding of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_field_type_embed', default=False, action='store_true',
                            help='Do not use embedding of the ASDL type of parent field to update decoder LSTM state')
    arg_parser.add_argument('--no_parent_state', default=False, action='store_true',
                            help='Do not use the parent hidden state to update decoder LSTM state')

    arg_parser.add_argument('--no_input_feed', default=False, action='store_true', help='Do not use input feeding in decoder LSTM')
    arg_parser.add_argument('--no_copy', default=False, action='store_true', help='Do not use copy mechanism')

    # Model configuration parameters specific for wikisql
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine', help='How to perform attention over table columns')
    arg_parser.add_argument('--answer_prune', dest='answer_prune', action='store_true', help='Whether to use answer pruning [default: True]')
    arg_parser.set_defaults(answer_prune=True)
    arg_parser.add_argument('--no_answer_prune', dest='answer_prune', action='store_false', help='Do not use answer prunning')

    #### Training ####
    arg_parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument('--glove_embed_path', default=None, type=str, help='Path to pretrained Glove mebedding')

    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--dev_file', type=str, help='path to the dev source file')

    arg_parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    arg_parser.add_argument('--dropout', default=0., type=float, help='Dropout rate')
    arg_parser.add_argument('--word_dropout', default=0., type=float, help='Word dropout rate')
    arg_parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='Word dropout rate on decoder')

    # training schedule details
    arg_parser.add_argument('--valid_metric', default='acc', choices=['acc'],
                            help='Metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int, help='Perform validation every x epoch')
    arg_parser.add_argument('--log_every', default=10, type=int, help='Log training statistics every n iterations')

    arg_parser.add_argument('--save_to', default='model', type=str, help='Save trained model to')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true', help='Save all intermediate checkpoints')
    arg_parser.add_argument('--patience', default=5, type=int, help='Training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int, help='Stop training after x number of trials')
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='If specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true', help='Use glorot initialization')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='Clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int, help='Decay learning rate after x epoch')
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False, help='Whether to reset optimizer when loading the best checkpoint')
    arg_parser.add_argument('--verbose', action='store_true', default=False, help='Verbose mode')
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=False,
                            help='Only evaluate the top prediction in validation')

    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_time_step', default=100, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')
    arg_parser.add_argument('--sample_size', default=5, type=int, help='Sample size')
    arg_parser.add_argument('--test_file', type=str, help='Path to the test file')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='Save decoding results to file')

    #### reranking ####
    arg_parser.add_argument('--load_reconstruction_model', type=str, help='Load reconstruction model')
    arg_parser.add_argument('--test_decode_file', default=None, type=str, help='Decoding results on test set')
    arg_parser.add_argument('--dev_decode_file', default=None, type=str, help='Decoding results on dev set')

    #### self-training ####
    arg_parser.add_argument('--load_decode_results', default=None, type=str)
    arg_parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')
    arg_parser.add_argument('--unlabeled_file', type=str, help='Path to the training source file used in semi-supervised self-training')

    #### interactive mode ####
    arg_parser.add_argument('--dataset_name', default=None, type=str)

    return arg_parser


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def update_args(args):
    for action in arg_parser._actions:
        if isinstance(action, argparse._StoreAction) or isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
            if not hasattr(args, action.dest):
                setattr(args, action.dest, action.default)


def train(args):
    """Maximum Likelihood Estimation"""

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = TransitionSystem.get_class_by_lang(args.lang)(grammar)
    train_set = Dataset.from_bin_file(args.train_file)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))
    
    if args.lang == 'wikisql':
        # import additional packages for wikisql dataset
        from model.wikisql.dataset import WikiSqlExample, WikiSqlTable, TableColumn

    parser_cls = get_parser_class(args.lang)
    model = parser_cls(args, vocab, transition_system)
    model.train()
    if args.cuda: model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
    elif args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    # load pre-trained word embedding (optional)
    if args.glove_embed_path:
        print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(args.glove_embed_path)
        glove_embedding.load_to(model.src_embed, vocab.source)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            train_iter += 1
            optimizer.zero_grad()

            ret_val = model.score(batch_examples)
            loss = -ret_val[0]

            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
                    sup_att_loss_val = sup_att_loss.data[0]
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_acc = eval_results['accuracy']
                print('[Epoch %d] code generation accuracy=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start), file=sys.stderr)
                is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
                history_dev_scores.append(dev_acc)
        else:
            is_better = True

            if epoch > args.lr_decay_after_epoch:
                lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                print('decay learning rate to %f' % lr, file=sys.stderr)

                # set new lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience and epoch >= args.lr_decay_after_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
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
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def train_reconstruction_model(args):
    train_set = Dataset.from_bin_file(args.train_file)
    dev_set = Dataset.from_bin_file(args.dev_file)
    vocab = pickle.load(open(args.vocab))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = TransitionSystem.get_class_by_lang(args.lang)(grammar)

    model = Reconstructor(args, vocab, transition_system)
    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def evaluate_ppl():
        model.eval()
        cum_loss = 0.
        cum_tgt_words = 0.
        for batch in dev_set.batch_iter(args.batch_size):
            loss = -model.score(batch).sum()
            cum_loss += loss.data[0]
            cum_tgt_words += sum(len(e.src_sent) + 1 for e in batch)  # add ending </s>

        ppl = np.exp(cum_loss / cum_tgt_words)
        model.train()
        return ppl

    print('begin training decoder, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            train_iter += 1
            optimizer.zero_grad()

            loss = -model.score(batch_examples)
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


def self_training(args):
    """Perform self-training

    First load decoding results on disjoint data
    also load pre-trained model and perform supervised
    training on both existing training data and the
    decoded results
    """

    print('load pre-trained model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_state = params['state_dict']

    # transfer arguments
    saved_args.cuda = args.cuda
    saved_args.save_to = args.save_to
    saved_args.train_file = args.train_file
    saved_args.unlabeled_file = args.unlabeled_file
    saved_args.dev_file = args.dev_file
    saved_args.load_decode_results = args.load_decode_results
    args = saved_args

    update_args(args)

    model = Parser(saved_args, vocab, transition_system)
    model.load_state_dict(saved_state)

    if args.cuda: model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('load unlabeled data [%s]' % args.unlabeled_file, file=sys.stderr)
    unlabeled_data = Dataset.from_bin_file(args.unlabeled_file)

    print('load decoding results of unlabeled data [%s]' % args.load_decode_results, file=sys.stderr)
    decode_results = pickle.load(open(args.load_decode_results))

    labeled_data = Dataset.from_bin_file(args.train_file)
    dev_set = Dataset.from_bin_file(args.dev_file)

    print('Num. examples in unlabeled data: %d' % len(unlabeled_data), file=sys.stderr)
    assert len(unlabeled_data) == len(decode_results)
    self_train_examples = []
    for example, hyps in zip(unlabeled_data, decode_results):
        if hyps:
            hyp = hyps[0]
            sampled_example = Example(idx='self_train-%s' % example.idx,
                                      src_sent=example.src_sent,
                                      tgt_code=hyp.code,
                                      tgt_actions=hyp.action_infos,
                                      tgt_ast=hyp.tree)
            self_train_examples.append(sampled_example)
    print('Num. self training examples: %d, Num. labeled examples: %d' % (len(self_train_examples), len(labeled_data)),
          file=sys.stderr)

    train_set = Dataset(examples=labeled_data.examples + self_train_examples)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]

            train_iter += 1
            optimizer.zero_grad()

            loss = -model.score(batch_examples)
            # print(loss.data)
            loss_val = torch.sum(loss).data[0]
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
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
        eval_results = evaluation.evaluate(dev_set.examples, model, args, verbose=True)
        dev_acc = eval_results['accuracy']
        print('[Epoch %d] code generation accuracy=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start), file=sys.stderr)
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
        elif epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)
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


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_state = params['state_dict']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    update_args(saved_args)

    parser_cls = get_parser_class(saved_args.lang)
    parser = parser_cls(saved_args, vocab, transition_system)

    parser.load_state_dict(saved_state)

    if args.cuda: parser = parser.cuda()
    parser.eval()

    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))


def interactive_mode(args):
    """Interactive mode"""
    print('Start interactive mode', file=sys.stderr)

    parser = StandaloneParser('atis',
                                 'saved_models/atis/'
                                 'model.atis.sup.lstm.hidden200.embed128.action128.field32.type32.dropout0.3.lr_decay0.5.beam5.vocab.bin.train.bin.glorot.par_state_w_field_embed.seed0.bin',
                                 beam_size=5,
                                 cuda=False)

    while True:
        utterance = input('Query:').strip()
        hypotheses = parser.parse(utterance, debug=True)

        for hyp_id, hyp in enumerate(hypotheses):
            print('------------------ Hypothesis %d ------------------' % hyp_id)
            print(hyp.code)
            print(hyp.tree.to_string())
            print('Actions:')
            for action_t in hyp.action_infos:
                print(action_t.__repr__(True))


def train_reranker_and_test(args):
    print('load dataset [test %s], [dev %s]' % (args.test_file, args.dev_file), file=sys.stderr)
    test_set = Dataset.from_bin_file(args.test_file)
    dev_set = Dataset.from_bin_file(args.dev_file)

    # print('load parser from [%s]' % args.load_model, file=sys.stderr)
    # parser_saved_args = torch.load(args.load_model, map_location=lambda storage, loc: storage)['args']
    # set the correct domain from saved arg
    # args.lang = parser_saved_args.lang

    # parser = get_parser_class(parser_saved_args.lang).load(args.load_model, cuda=args.cuda)
    print('load reconstruction model from [%s]' % args.load_reconstruction_model, file=sys.stderr)
    reconstruction_model = Reconstructor.load(args.load_reconstruction_model, cuda=args.cuda)
    transition_system = reconstruction_model.transition_system

    # transition_system = parser.transition_system
    # parser.eval()

    def _filter_hyps(_decode_results):
        for i in range(len(_decode_results)):
            valid_hyps = []
            for hyp in _decode_results[i]:
                try: 
                    transition_system.tokenize_code(hyp.code)
                    valid_hyps.append(hyp)
                except: pass

            _decode_results[i] = valid_hyps

    print('load dev decode results [%s]' % args.dev_decode_file, file=sys.stderr)
    dev_decode_results = pickle.load(open(args.dev_decode_file, 'rb'))
    _filter_hyps(dev_decode_results)
    beam_size = max(len(hyps) for hyps in dev_decode_results)
    dev_acc = sum(hyps and hyps[0].correct for hyps in dev_decode_results) / float(len(dev_set))
    dev_oracle = sum(any(hyp.correct for hyp in hyps) for hyps in dev_decode_results) / float(len(dev_set))

    print('load test decode results [%s]' % args.test_decode_file, file=sys.stderr)
    test_decode_results = pickle.load(open(args.test_decode_file, 'rb'))
    _filter_hyps(test_decode_results)
    test_acc = sum(hyps and hyps[0].correct for hyps in test_decode_results) / float(len(test_set))
    test_oracle = sum(any(hyp.correct for hyp in hyps) for hyps in test_decode_results) / float(len(test_set))

    print('Dev Acc@1=%.4f, Test Oracle Acc=%.4f' % (dev_acc, dev_oracle), file=sys.stderr)

    def _compute_rerank_feature(_examples, _decode_results):
        hyp_examples = []

        for example, hyps in zip(_examples, _decode_results):
            for hyp in hyps:
                hyp_example = Example(idx=None,
                                      src_sent=example.src_sent,
                                      tgt_code=hyp.code,
                                      tgt_actions=None,
                                      tgt_ast=None)
                hyp_examples.append(hyp_example)

        for batch_examples in utils.batch_iter(hyp_examples, batch_size=128):
            batch_example_scores = reconstruction_model.score(batch_examples).data.cpu().tolist()
            for i, e in enumerate(batch_examples):
                e.reconstruction_score = batch_example_scores[i]

        e_ptr = 0
        for example, hyps in zip(_examples, _decode_results):
            for hyp in hyps:
                hyp.reconstruction_score = hyp_examples[e_ptr].reconstruction_score
                e_ptr += 1

    def _compute_rerank_performance(_examples, _decode_results, _eta):
        correct_array = []
        for example, hyps in zip(_examples, _decode_results):
            if hyps:
                best_hyp_idx = np.argmax([hyp.score + _eta * hyp.reconstruction_score for hyp in hyps])
                is_correct = hyps[best_hyp_idx].correct
                correct_array.append(is_correct)

        acc = sum(correct_array) / float(len(_examples))
        return acc

    best_eta = eta = 0.
    delta_eta = 0.01
    best_dev_score = dev_acc

    _compute_rerank_feature(dev_set.examples, dev_decode_results)
    while eta <= 1.0:
        eta += delta_eta

        # compute new dev score using current eta
        dev_score = _compute_rerank_performance(dev_set.examples, dev_decode_results, eta)
        if dev_score > best_dev_score:
            print('New eta=%.4f, dev score=%.4f' % (eta, dev_score), file=sys.stderr)
            best_eta = eta
            best_dev_score = dev_score

    # test!
    _compute_rerank_feature(test_set.examples, test_decode_results)
    test_score_with_rerank = _compute_rerank_performance(test_set.examples, test_decode_results, best_eta)

    print('Test Acc@1=%.4f, Test Re-rank Acc=%.4f, Test Oracle Acc=%.4f' % (test_acc,
                                                                            test_score_with_rerank,
                                                                            test_oracle), file=sys.stderr)


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'self_train':
        self_training(args)
    elif args.mode == 'train_reconstructor':
        train_reconstruction_model(args)
    elif args.mode == 'rerank':
        train_reranker_and_test(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'interactive':
        interactive_mode(args)
    else:
        raise RuntimeError('unknown mode')
