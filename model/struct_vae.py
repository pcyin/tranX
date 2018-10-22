# coding=utf-8
from __future__ import print_function

import sys
import traceback

import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from components.dataset import Example
from model.prior import UniformPrior
from parser import *
from model.reconstruction_model import *


class StructVAE(nn.Module):
    def __init__(self, encoder, decoder, prior, args):
        super(StructVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.transition_system = self.encoder.transition_system
        self.args = args

        # for baseline
        self.b_x_l1 = nn.Linear(args.hidden_size, 20)
        self.b_x_l2 = nn.Linear(20, 1, bias=False)
        self.b = nn.Parameter(torch.FloatTensor(1))

        # initialize baseline to be a small negative number
        self.b.data.fill_(-20.)

    def get_unsupervised_loss(self, examples):
        samples, sample_scores, enc_states = self.infer(examples)

        reconstruction_scores = self.decoder.score(samples)

        # compute prior probability
        prior_scores = self.prior([e.tgt_code for e in samples])
        if isinstance(self.prior, UniformPrior):
            prior_scores = Variable(sample_scores.data.new(prior_scores))

        kl_term = self.args.alpha * (sample_scores - prior_scores)
        raw_learning_signal = reconstruction_scores - kl_term
        baseline = self.baseline(samples, enc_states)
        learning_signal = raw_learning_signal.detach() - baseline

        # clip learning signal
        if self.args.clip_learning_signal is not None:
            mask = torch.lt(learning_signal, self.args.clip_learning_signal).float()
            clipped_learning_signal = learning_signal * (1. - mask) + mask * self.args.clip_learning_signal
        else:
            clipped_learning_signal = learning_signal

        encoder_loss = -clipped_learning_signal.detach() * sample_scores
        decoder_loss = -reconstruction_scores

        # compute baseline loss
        baseline_loss = learning_signal ** 2

        meta_data = {'samples': samples,
                     'reconstruction_scores': reconstruction_scores,
                     'encoding_scores': sample_scores,
                     'raw_learning_signal': raw_learning_signal,
                     'learning_signal': learning_signal,
                     'baseline': baseline,
                     'kl_term': kl_term,
                     'prior': prior_scores}

        return encoder_loss, decoder_loss, baseline_loss, meta_data

    def baseline(self, samples, enc_states):
        # compute baseline, which is an MLP
        # (sample_size) FIXME: reward is log-likelihood, shall we use activation here?

        b_x = self.b_x_l2(F.tanh(self.b_x_l1(enc_states.detach()))).view(-1)

        return b_x + self.b

    def infer(self, examples):
        # currently use beam search as sampling method
        # set model to evaluation model for beam search, make sure dropout is properly behaving!
        was_training = self.encoder.training
        self.encoder.eval()

        hypotheses = [self.encoder.parse(e.src_sent, beam_size=self.args.sample_size) for e in examples]

        if len(hypotheses) == 0:
            raise ValueError('No candidate hypotheses.')

        if was_training: self.encoder.train()

        # some source may not have corresponding samples, so we only retain those that have sampled logical forms
        sampled_examples = []
        for e_id, (example, hyps) in enumerate(zip(examples, hypotheses)):
            for hyp_id, hyp in enumerate(hyps):
                try:
                    code = self.transition_system.ast_to_surface_code(hyp.tree)
                    self.transition_system.tokenize_code(code)  # make sure the code is tokenizable!
                    sampled_example = Example(idx='%d-sample%d' % (example.idx, hyp_id),
                                              src_sent=example.src_sent,
                                              tgt_code=code,
                                              tgt_actions=hyp.action_infos,
                                              tgt_ast=hyp.tree)
                    sampled_examples.append(sampled_example)
                except:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        sample_scores, enc_states = self.encoder.score(sampled_examples, return_encode_state=True)

        return sampled_examples, sample_scores, enc_states

    def save(self, path):
        fname, ext = os.path.splitext(path)
        self.encoder.save(fname + '.encoder' + ext)
        self.decoder.save(fname + '.decoder' + ext)
        state_dict = {k: v for k, v in self.state_dict().iteritems() if not (k.startswith('decoder') or k.startswith('encoder') or k.startswith('prior'))}

        params = {
            'args': self.args,
            'state_dict': state_dict
        }

        torch.save(params, path)

    def load_parameters(self, path):
        fname, ext = os.path.splitext(path)
        encoder_states = torch.load(fname + '.encoder' + ext, map_location=lambda storage, loc: storage)['state_dict']
        self.encoder.load_state_dict(encoder_states)

        decoder_states = torch.load(fname + '.decoder' + ext, map_location=lambda storage, loc: storage)['state_dict']
        self.decoder.load_state_dict(decoder_states)

        vae_states = torch.load(path, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(vae_states, strict=False)

    def train(self):
        super(StructVAE, self).train()
        self.prior.eval()


class StructVAE_LMBaseline(StructVAE):
    def __init__(self, encoder, decoder, prior, src_lm, args):
        super(StructVAE_LMBaseline, self).__init__(encoder, decoder, prior, args)

        del self.b_x_l1
        del self.b_x_l2

        self.b_lm = src_lm
        self.b_lm_weight = nn.Parameter(torch.FloatTensor([.5 if args.lang == 'python' else 0.9]))  # FIXME: language dependant init

        # initialize baseline to be a small negative number
        self.b.data.fill_(-2. if args.lang == 'python' else 2.)

    def baseline(self, samples, enc_states):
        src_sent_var = nn_utils.to_input_variable([e.src_sent for e in samples],
                                                  self.b_lm.vocab, cuda=self.args.cuda,
                                                  append_boundary_sym=True)
        p_lm = -self.b_lm(src_sent_var)

        return self.b_lm_weight * p_lm - self.b
    
    def train(self):
        super(StructVAE_LMBaseline, self).train()
        self.b_lm.eval()


class StructVAE_SrcLmAndLinearBaseline(StructVAE_LMBaseline):
    def __init__(self, encoder, decoder, prior, src_lm, args):
        super(StructVAE_SrcLmAndLinearBaseline, self).__init__(encoder, decoder, prior, src_lm, args)

        # For MLP baseline
        # for baseline
        self.b_x_l1 = nn.Linear(args.hidden_size, 1)

    def baseline(self, samples, enc_states):
        b_linear = self.b_x_l1(enc_states.detach()).view(-1)

        b_lm = super(StructVAE_SrcLmAndLinearBaseline, self).baseline(samples, enc_states)

        return b_linear + b_lm