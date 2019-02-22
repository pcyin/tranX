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
from components.reranker import *
from components.standalone_parser import StandaloneParser
from common.utils import update_args, init_arg_parser
from model import nn_utils, utils
from model.neural_lm import LSTMLanguageModel
from model.paraphrase import ParaphraseIdentificationModel

from model.parser import Parser
from model.prior import UniformPrior, LSTMPrior
from model.reconstruction_model import Reconstructor
from model.struct_vae import StructVAE, StructVAE_LMBaseline, StructVAE_SrcLmAndLinearBaseline
from model.utils import GloveHelper, get_parser_class


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


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
    evaluator = Registrable.by_name(args.evaluator)(transition_system)
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
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                    epoch, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start), file=sys.stderr)

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
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


def train_rerank_feature(args):
    train_set = Dataset.from_bin_file(args.train_file)
    dev_set = Dataset.from_bin_file(args.dev_file)
    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = TransitionSystem.get_class_by_lang(args.lang)(grammar)

    train_paraphrase_model = args.mode == 'train_paraphrase_identifier'

    def _get_feat_class():
        if args.mode == 'train_reconstructor':
            return Reconstructor
        elif args.mode == 'train_paraphrase_identifier':
            return ParaphraseIdentificationModel

    def _filter_hyps(_decode_results):
        for i in range(len(_decode_results)):
            valid_hyps = []
            for hyp in _decode_results[i]:
                try:
                    transition_system.tokenize_code(hyp.code)
                    valid_hyps.append(hyp)
                except: pass

            _decode_results[i] = valid_hyps

    model = _get_feat_class()(args, vocab, transition_system)

    if args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    model.train()
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # if training the paraphrase model, also load in decoding results
    if train_paraphrase_model:
        print('load training decode results [%s]' % args.train_decode_file, file=sys.stderr)
        train_decode_results = pickle.load(open(args.train_decode_file, 'rb'))
        _filter_hyps(train_decode_results)
        train_decode_results = {e.idx: hyps for e, hyps in zip(train_set, train_decode_results)}

        print('load dev decode results [%s]' % args.dev_decode_file, file=sys.stderr)
        dev_decode_results = pickle.load(open(args.dev_decode_file, 'rb'))
        _filter_hyps(dev_decode_results)
        dev_decode_results = {e.idx: hyps for e, hyps in zip(dev_set, dev_decode_results)}

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

    def evaluate_paraphrase_acc():
        model.eval()
        labels = []
        for batch in dev_set.batch_iter(args.batch_size):
            probs = model.score(batch).exp().data.cpu().numpy()
            for p in probs:
                labels.append(p >= 0.5)

            # get negative examples
            batch_decoding_results = [dev_decode_results[e.idx] for e in batch]
            batch_negative_examples = [get_negative_example(e, _hyps, type='best')
                                       for e, _hyps in zip(batch, batch_decoding_results)]
            batch_negative_examples = list(filter(None, batch_negative_examples))
            probs = model.score(batch_negative_examples).exp().data.cpu().numpy()
            for p in probs:
                labels.append(p < 0.5)

        acc = np.average(labels)
        model.train()
        return acc

    def get_negative_example(_example, _hyps, type='sample'):
        if _hyps:
            incorrect_hyps = [hyp for hyp in _hyps if not hyp.is_correct]
            incorrect_hyp_scores = [hyp.score for hyp in incorrect_hyps]
            if type in ('best', 'sample'):
                if type == 'best':
                    sample_idx = np.argmax(incorrect_hyp_scores)
                    sampled_hyp = incorrect_hyps[sample_idx]
                else:
                    incorrect_hyp_probs = [np.exp(score) for score in incorrect_hyp_scores]
                    incorrect_hyp_probs = np.array(incorrect_hyp_probs) / sum(incorrect_hyp_probs)
                    sampled_hyp = np.random.choice(incorrect_hyps, size=1, p=incorrect_hyp_probs)
                    sampled_hyp = sampled_hyp[0]

                sample = Example(idx='negative-%s' % _example.idx,
                                 src_sent=_example.src_sent,
                                 tgt_code=sampled_hyp.code,
                                 tgt_actions=None,
                                 tgt_ast=None)
                return sample
            elif type == 'all':
                samples = []
                for i, hyp in enumerate(incorrect_hyps):
                    sample = Example(idx='negative-%s-%d' % (_example.idx, i),
                                     src_sent=_example.src_sent,
                                     tgt_code=hyp.code,
                                     tgt_actions=None,
                                     tgt_ast=None)
                    samples.append(sample)

                return samples
        else:
            return None

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

            if train_paraphrase_model:
                positive_examples_num = len(batch_examples)
                labels = [0] * len(batch_examples)
                negative_samples = []
                batch_decoding_results = [train_decode_results[e.idx] for e in batch_examples]
                # sample negative examples
                for example, hyps in zip(batch_examples, batch_decoding_results):
                    if hyps:
                        negative_sample = get_negative_example(example, hyps, type=args.negative_sample_type)
                        if isinstance(negative_sample, Example):
                            negative_samples.append(negative_sample)
                            labels.append(1)
                        else:
                            negative_samples.extend(negative_sample)
                            labels.extend([1] * len(negative_sample))

                batch_examples += negative_samples

            train_iter += 1
            optimizer.zero_grad()

            nll = -model(batch_examples)
            if train_paraphrase_model:
                idx_tensor = Variable(torch.LongTensor(labels).unsqueeze(-1), requires_grad=False)
                if args.cuda: idx_tensor = idx_tensor.cuda()
                loss = torch.gather(nll, 1, idx_tensor)
            else:
                loss = nll

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
        # evaluate dev_score
        dev_acc = evaluate_paraphrase_acc() if train_paraphrase_model else -evaluate_ppl()
        print('[Epoch %d] dev_score=%.5f took %ds' % (epoch, dev_acc, time.time() - eval_start), file=sys.stderr)
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

    update_args(args, arg_parser)

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
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = get_parser_class(saved_args.lang)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(transition_system)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))


def interactive_mode(args):
    """Interactive mode"""
    print('Start interactive mode', file=sys.stderr)

    parser = StandaloneParser('conala',
                              'saved_models/conala/'
                              'model.sup.conala.lstm.hidden256.embed128.action128.field64.type64.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.vocab.var_str_sep.new_dev.src_freq3.code_freq3.bin.train.var_str_sep.new_dev.bin.glorot.par_state.seed1.bin',
                              beam_size=15,
                              cuda=False)

    while True:
        utterance = input('Query:').strip()
        hypotheses = parser.parse(utterance, debug=True)

        for hyp_id, hyp in enumerate(hypotheses):
            print('------------------ Hypothesis %d ------------------' % hyp_id)
            print(hyp.code)
            # print(hyp.tree.to_string())
            # print('Actions:')
            # for action_t in hyp.action_infos:
            #     print(action_t.__repr__(True))


def train_reranker_and_test(args):
    print('load dataset [test %s], [dev %s]' % (args.test_file, args.dev_file), file=sys.stderr)
    test_set = Dataset.from_bin_file(args.test_file)
    dev_set = Dataset.from_bin_file(args.dev_file)

    features = []
    i = 0
    while i < len(args.features):
        feat_name = args.features[i]
        feat_cls = Registrable.by_name(feat_name)
        print('Add feature %s' % feat_name, file=sys.stderr)
        if issubclass(feat_cls, nn.Module):
            i += 1
            feat_path = args.features[i]
            feat_inst = feat_cls.load(feat_path)
            print('Load feature %s from %s' % (feat_name, feat_path), file=sys.stderr)
        else:
            feat_inst = feat_cls()

        features.append(feat_inst)
        i += 1

    transition_system = next(feat.transition_system for feat in features if hasattr(feat, 'transition_system'))
    evaluator = Registrable.by_name(args.evaluator)(transition_system)

    reranker = GridSearchReranker(features, transition_system=transition_system)

    print('load dev decode results [%s]' % args.dev_decode_file, file=sys.stderr)
    dev_decode_results = pickle.load(open(args.dev_decode_file, 'rb'))
    dev_eval_results = evaluator.evaluate_dataset(dev_set, dev_decode_results, fast_mode=False)

    print('load test decode results [%s]' % args.test_decode_file, file=sys.stderr)
    test_decode_results = pickle.load(open(args.test_decode_file, 'rb'))
    test_eval_results = evaluator.evaluate_dataset(test_set, test_decode_results, fast_mode=False)

    print('Dev Eval Results', file=sys.stderr)
    print(dev_eval_results, file=sys.stderr)
    print('Test Eval Results', file=sys.stderr)
    print(test_eval_results, file=sys.stderr)

    if args.num_workers == 1:
        reranker.train(dev_set.examples, dev_decode_results, evaluator=evaluator)
    else:
        reranker.train_multiprocess(dev_set.examples, dev_decode_results, evaluator=evaluator, num_workers=args.num_workers)

    test_score_with_rerank = reranker.compute_rerank_performance(test_set.examples, test_decode_results, verbose=True,
                                                                 evaluator=evaluator)

    print('Test Eval Results After Reranking', file=sys.stderr)
    print(test_score_with_rerank, file=sys.stderr)

    if args.save_to:
        print('Save Reranker to %s' % args.save_to, file=sys.stderr)
        reranker.save(args.save_to)


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'self_train':
        self_training(args)
    elif args.mode in ('train_reconstructor', 'train_paraphrase_identifier'):
        train_rerank_feature(args)
    elif args.mode == 'rerank':
        train_reranker_and_test(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'interactive':
        interactive_mode(args)
    else:
        raise RuntimeError('unknown mode')
