# coding=utf-8
from __future__ import print_function

import itertools
import sys
from collections import OrderedDict

import numpy as np

from datasets.conala.conala_eval import tokenize_for_bleu_eval
from datasets.conala import evaluator as conala_evaluator
from model import utils
from components.dataset import Example

import xgboost as xgb
from sklearn import preprocessing


class RerankingFeature(object):
    @property
    def feature_name(self):
        raise NotImplementedError

    @property
    def is_batched(self):
        raise NotImplementedError

    def get_feat_value(self, example, hyp, **kwargs):
        raise NotImplementedError


class IsSecondHypAndScoreMargin(RerankingFeature):
    def __init__(self):
        pass

    @property
    def feature_name(self):
        return 'is_2nd_hyp_and_margin_with_top_hyp'

    @property
    def is_batched(self):
        return False

    def get_feat_value(self, example, hyp, **kwargs):
        if kwargs['hyp_id'] == 1:
            return kwargs['all_hyps'][0].score - hyp.score
        return 0.


class IsSecondHypAndParaphraseScoreMargin(RerankingFeature):
    def __init__(self):
        pass

    @property
    def feature_name(self):
        return 'is_2nd_hyp_and_paraphrase_score_margin_with_top_hyp'

    @property
    def is_batched(self):
        return False

    def get_feat_value(self, example, hyp, **kwargs):
        if kwargs['hyp_id'] == 1:
            return hyp.rerank_feature_values['paraphrase_score'] - kwargs['all_hyps'][0].rerank_feature_values['paraphrase_score']
        return 0.


class Reranker(object):
    def __init__(self, features, parameter=None):
        self.features = []
        self.feat_map = OrderedDict()
        self.batched_features = OrderedDict()

        for feat in features:
            self._add_feature(feat)

        if parameter is not None:
            self.parameter = np.array(parameter)
        else:
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
        raise NotImplementedError

    def initialize_rerank_features(self, examples, decode_results, metric='accuracy'):
        hyp_examples = []
        for example, hyps in zip(examples, decode_results):
            for hyp_id, hyp in enumerate(hyps):
                hyp_example = Example(idx=None,
                                      src_sent=example.src_sent,
                                      tgt_code=hyp.code,
                                      tgt_actions=None,
                                      tgt_ast=None)
                hyp_examples.append(hyp_example)

                feat_vals = OrderedDict()
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

        for example, hyps in zip(examples, decode_results):
            for hyp_id, hyp in enumerate(hyps):
                for feat_name, feat in self.feat_map.items():
                    if not feat.is_batched:
                        feat_val = feat.get_feat_value(example, hyp, hyp_id=hyp_id, all_hyps=hyps)
                        hyp.rerank_feature_values[feat_name] = feat_val

        # also pre-tokenize references and hypotheses for BLEU evaluation
        if metric == 'bleu':
            for example in examples:
                ref_code_tokens = tokenize_for_bleu_eval(example.meta['example_dict']['snippet'])
                example.meta['example_dict']['ref_code_tokens'] = ref_code_tokens
            for hyp_list in decode_results:
                for hyp in hyp_list:
                    hyp.decanonical_code_tokens = tokenize_for_bleu_eval(hyp.decanonical_code)

    def get_rerank_score(self, hyp, param):
        raise NotImplementedError

    def compute_rerank_performance(self, examples, decode_results, param=None, verbose=False, metric='accuracy', full_metric=False):
        if not hasattr(decode_results[0][0], 'rerank_feature_values'):
            print('initializing rerank features for hypotheses...', file=sys.stderr)
            self.initialize_rerank_features(examples, decode_results, metric=metric)

        if param is None:
            param = self.parameter

        correct_array = []
        hypotheses = []
        sorted_decode_results = []
        for example, hyps in zip(examples, decode_results):
            if hyps:
                best_hyp_idx = np.argmax([self.get_rerank_score(hyp, param=param) for hyp in hyps])
                best_hyp = hyps[best_hyp_idx]
                is_correct = best_hyp.correct

                if metric == 'bleu':
                    hypotheses.append(best_hyp.decanonical_code_tokens)

                if full_metric:
                    sorted_decode_results.append([best_hyp])
            else:
                is_correct = False

                if metric == 'bleu':
                    hypotheses.append('')

                if full_metric:
                    sorted_decode_results.append([])

            correct_array.append(is_correct)
            if verbose:
                gold_standard_idx = [i for i, hyp in enumerate(hyps) if hyp.correct]
                if not is_correct and gold_standard_idx:
                    gold_standard_idx = gold_standard_idx[0]
                    print('Utterance: %s' % ' '.join(example.src_sent), file=sys.stderr)
                    print('Gold hyp id: %d' % gold_standard_idx, file=sys.stderr)
                    for _i, hyp in enumerate(hyps):
                        print('Hyp %d: %s ||| score: %f ||| final score: %f' % (_i,
                                                                                hyp.code,
                                                                                hyp.score,
                                                                                self.get_rerank_score(hyp, param=param)),
                              file=sys.stderr)
                        print('\t%s' % hyp.rerank_feature_values, file=sys.stderr)

        acc = np.average(correct_array)
        if metric == 'bleu':
            if full_metric:
                eval_results = conala_evaluator.evaluate(examples, sorted_decode_results)
                return eval_results
            else:
                references = [e.meta['example_dict']['ref_code_tokens'] for e in examples]
                assert len(references) == len(hypotheses)
                corpus_bleu = conala_evaluator.evaluate(references=references, hypotheses=hypotheses)
                return corpus_bleu

        return acc

    def train(self, examples, decode_results, initial_performance=0., metric='accuracy'):
        raise NotImplementedError

    @property
    def feature_num(self):
        return len(self.features)

    def __getattr__(self, item):
        if item in self.feat_map:
            return self.feat_map.get(item)
        raise ValueError


class MERTReranker(Reranker):
    """MERT reranker"""

    def get_rerank_score(self, hyp, param):
        feat_vals = np.array(list(hyp.rerank_feature_values.values()))
        score = hyp.score + np.dot(param, feat_vals)

        return score

    def train(self, examples, decode_results, initial_performance=0., metric='accuracy'):
        """optimize the ranker on a dataset using grid search"""
        best_score = initial_performance
        best_param = np.zeros(self.feature_num)

        param_space = (np.array(p) for p in itertools.combinations(np.arange(0, 1.01, 0.01), self.feature_num))

        for param in param_space:
            score = self.compute_rerank_performance(examples, decode_results, param=param, metric=metric)
            if score > best_score:
                print('New param=%s, score=%.4f' % (param, score), file=sys.stderr)
                best_param = param
                best_score = score

        self.parameter = best_param


class XGBoostReranker(Reranker):
    def __init__(self, features):
        super(XGBoostReranker, self).__init__(features)

        params = {'objective': 'rank:ndcg', 'learning_rate': .3,
                  'gamma': 1.0, 'min_child_weight': 0.1,
                  'max_depth': 4, 'n_estimators': 5}

        self.ranker = xgb.sklearn.XGBRanker(**params)

    def get_feature_matrix(self, decode_results, train=False):
        x, y, group = [], [], []

        for hyps in decode_results:
            if hyps:
                for hyp in hyps:
                    label = 1 if hyp.correct else 0
                    feat_vec = np.array([hyp.score] + [v for v in hyp.rerank_feature_values.values()])
                    x.append(feat_vec)
                    y.append(label)
                group.append(len(hyps))

        x = np.stack(x)
        y = np.array(y)

        # if train:
        #     self.scaler = preprocessing.StandardScaler().fit(x)
        #     x = self.scaler.transform(x)
        # else:
        #     x = self.scaler.transform(x)

        return x, y, group

    def get_rerank_score(self, hyp, param):
        x, y, group = self.get_feature_matrix([[hyp]])
        y = self.ranker.predict(x)

        return y[0]

    def train(self, examples, decode_results, initial_performance=0.):
        self.initialize_rerank_features(examples, decode_results)

        train_x, train_y, group_train = self.get_feature_matrix(decode_results, train=True)
        self.ranker.fit(train_x, train_y, group_train)

        train_acc = self.compute_rerank_performance(examples, decode_results)
        print('Dev acc: %f' % train_acc, file=sys.stderr)
