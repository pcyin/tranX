# coding=utf-8
from __future__ import print_function

import itertools
import sys
from collections import OrderedDict

import numpy as np

from model import utils
from components.dataset import Example


class RerankingFeature(object):
    @property
    def feature_name(self):
        raise NotImplementedError

    @property
    def is_batched(self):
        raise NotImplementedError

    def get_feat_value(self, example, hyp, **kwargs):
        raise NotImplementedError


class IsSecondHypAndNegativeScoreMargin(RerankingFeature):
    def __init__(self):
        pass

    @property
    def feature_name(self):
        return 'is_2nd_hyp_and_margin_with_top_hyp'

    @property
    def is_batched(self):
        return False

    def get_feat_value(self, example, hyp, **kwargs):
        if kwargs['hyp_id'] == 1 and (kwargs['all_hyps'][0].score - hyp.score) <= 1.:
            return 1.
        return 0.


class Reranker(object):
    def __init__(self, features):
        self.features = []
        self.feat_map = OrderedDict()
        self.batched_features = OrderedDict()

        for feat in features:
            self._add_feature(feat)

        self.parameter = np.zeros(self.feature_num)

    def _add_feature(self, feature):
        self.features.append(feature)
        self.feat_map[feature.feature_name] = feature

        if feature.is_batched:
            self.batched_features[feature.feature_name] = feature

    def get_initial_reranking_feature_values(self, example, hyp, **kwargs):
        """Given a hypothesis, compute its reranking feature"""
        feat_values = OrderedDict()
        for feat_name, feat in self.feat_map.items():
            if not feat.is_batched:
                feat_val = feat.get_feat_value(example, hyp, **kwargs)
            else:
                feat_val = float('inf')

            feat_values[feat_name] = feat_val

        return feat_values

    def rerank_hypotheses(self, example, hypotheses):
        """rerank the hypotheses using the current model parameter"""
        pass

    def initialize_rerank_features(self, examples, decode_results):
        hyp_examples = []
        for example, hyps in zip(examples, decode_results):
            for hyp_id, hyp in enumerate(hyps):
                hyp_example = Example(idx=None,
                                      src_sent=example.src_sent,
                                      tgt_code=hyp.code,
                                      tgt_actions=None,
                                      tgt_ast=None)
                hyp_examples.append(hyp_example)

                feat_vals = self.get_initial_reranking_feature_values(example, hyp, hyp_id=hyp_id, all_hyps=hyps)
                hyp.rerank_feature_values = feat_vals

        for batch_examples in utils.batch_iter(hyp_examples, batch_size=128):
            for feat_name, feat in self.batched_features.items():
                batch_example_scores = feat.score(batch_examples).data.cpu().tolist()
                for i, e in enumerate(batch_examples):
                    setattr(e, feat_name, batch_example_scores[i])

        e_ptr = 0
        for example, hyps in zip(examples, decode_results):
            for hyp in hyps:
                for feat_name, feat in self.batched_features.items():
                    hyp.rerank_feature_values[feat_name] = getattr(hyp_examples[e_ptr], feat_name)
                e_ptr += 1

    @staticmethod
    def get_rerank_score(hyp, param):
        feat_vals = np.array(list(hyp.rerank_feature_values.values()))
        score = hyp.score + np.dot(param, feat_vals)

        return score

    def compute_rerank_performance(self, examples, decode_results, param=None):
        if not hasattr(decode_results[0][0], 'rerank_feature_values'):
            print('initializing rerank features for hypotheses...', file=sys.stderr)
            self.initialize_rerank_features(examples, decode_results)

        if param is None:
            param = self.parameter

        correct_array = []
        for example, hyps in zip(examples, decode_results):
            if hyps:
                best_hyp_idx = np.argmax([self.get_rerank_score(hyp, param=param) for hyp in hyps])
                is_correct = hyps[best_hyp_idx].correct
                correct_array.append(is_correct)
            else:
                correct_array.append(False)

        acc = np.average(correct_array)
        return acc

    def train(self, examples, decode_results, initial_performance=0.):
        """optimize the ranker on a dataset using grid search"""
        best_score = initial_performance
        best_param = np.zeros(self.feature_num)

        param_space = (np.array(p) for p in itertools.combinations(np.arange(0, 1.01, 0.01), self.feature_num))

        for param in param_space:
            score = self.compute_rerank_performance(examples, decode_results, param=param)
            if score > best_score:
                print('New param=%s, score=%.4f' % (param, score), file=sys.stderr)
                best_param = param
                best_score = score

        self.parameter = best_param

    @property
    def feature_num(self):
        return len(self.features)

    def __getattr__(self, item):
        if item in self.feat_map:
            return self.feat_map.get(item)
        raise ValueError
