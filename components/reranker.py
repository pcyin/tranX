# coding=utf-8
from __future__ import print_function
import multiprocessing
import math
import itertools
import re
import sys, os
import six
import pickle
import torch
import torch.nn as nn
from collections import OrderedDict

import numpy as np

from components.evaluator import CachedExactMatchEvaluator
from common.registerable import Registrable
from common.savable import Savable
from datasets.conala.conala_eval import tokenize_for_bleu_eval
from datasets.conala import evaluator as conala_evaluator
from model import utils
from components.dataset import Example, Dataset

if six.PY3:
    from asdl import Python3TransitionSystem
else:
    from asdl import PythonTransitionSystem

import xgboost as xgb


# shared across processes for multi-processed reranking
_examples = None
_decode_results = None
_evaluator = None
_ranker = None


def _rank_worker(param):
    score = _ranker.compute_rerank_performance(_examples, _decode_results, fast_mode=True, evaluator=_evaluator, param=param)
    return param, score


def _rank_segment_worker(param_space):
    best_score = 0.
    best_param = None
    print('[Child] New parameter segments [%s ~ %s] (%d entries)' % (param_space[0], param_space[-1], len(param_space)), file=sys.stderr)
    for param in param_space:
        score = _ranker.compute_rerank_performance(_examples, _decode_results, fast_mode=True, evaluator=_evaluator, param=np.array(param))
        if score > best_score:
            print('[Child] New param=%s, score=%.4f' % (param, score), file=sys.stderr)
            best_param = param
            best_score = score

    return best_param, best_score


class RerankingFeature(object):
    @property
    def feature_name(self):
        raise NotImplementedError

    @property
    def is_batched(self):
        raise NotImplementedError

    def get_feat_value(self, example, hyp, **kwargs):
        raise NotImplementedError


@Registrable.register('parser_score')
class ParserScore(RerankingFeature):
    @property
    def feature_name(self):
        return 'parser_score'

    @property
    def is_batched(self):
        return False

    def get_feat_value(self, example, hyp, **kwargs):
        return float(hyp.score)


@Registrable.register('normalized_parser_score_by_action')
class NormalizedParserScoreByAction(RerankingFeature):
    def __init__(self):
        pass

    @property
    def feature_name(self):
        return 'normalized_parser_score'

    @property
    def is_batched(self):
        return False

    def get_feat_value(self, example, hyp, **kwargs):
        # return float(hyp.score) / len(kwargs['transition_system'].tokenize_code(hyp.code))
        return float(hyp.score) / len(hyp.actions)


@Registrable.register('normalized_parser_score')
class NormalizedParserScore(RerankingFeature):
    def __init__(self):
        pass

    @property
    def feature_name(self):
        return 'normalized_parser_score'

    @property
    def is_batched(self):
        return False

    def get_feat_value(self, example, hyp, **kwargs):
        return float(hyp.score) / HypCodeTokensCount().get_feat_value(example, hyp, **kwargs)


@Registrable.register('word_cnt')
class HypCodeTokensCount(RerankingFeature):
    @property
    def feature_name(self):
        return 'word_cnt'

    @property
    def is_batched(self):
        return False

    def get_feat_value(self, example, hyp, **kwargs):
        # TODO: this is dataset specific, we should fix it
        if hasattr(hyp, 'decanonical_code_tokens'):
            # we use the tokenization for BLEU calculation
            code_tokens = ['#NEWLINE#' if c == '\n' else c for c in hyp.decanonical_code_tokens]
            return float(len(code_tokens))
        # else:
        #     return len(hyp.actions)
        elif six.PY2 and isinstance(kwargs['transition_system'], PythonTransitionSystem):
            code_tokens = [c.replace('\r\n', '#NEWLINE#').replace('\r', '#NEWLINE#').replace('\n', '#NEWLINE#')
                           for c in kwargs['transition_system'].tokenize_code(hyp.code)]
            # remove consecutive spaces
            code_tokens = re.sub(r'\s+', ' ', ' '.join(code_tokens)).strip().split(' ')
            code_tokens = list(filter(lambda x: len(x) > 0, code_tokens))
            return float(len(code_tokens))

        return len(kwargs['transition_system'].tokenize_code(hyp.code))


@Registrable.register('is_2nd_hyp_and_margin_with_top_hyp')
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


@Registrable.register('is_2nd_hyp_and_paraphrase_score_margin_with_top_hyp')
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


@Registrable.register('reranker')
class Reranker(Savable):
    def __init__(self, features, parameter=None, transition_system=None):
        self.features = []
        self.transition_system = transition_system
        self.feat_map = OrderedDict()
        self.batched_features = OrderedDict()

        for feat in features:
            self._add_feature(feat)

        if parameter is not None:
            self.parameter = parameter
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

    def initialize_rerank_features(self, examples, decode_results):
        hyp_examples = []
        print('initializing features...', file=sys.stderr)
        for example, hyps in zip(examples, decode_results):
            for hyp_id, hyp in enumerate(hyps):
                hyp_example = Example(idx=None,
                                      src_sent=example.src_sent,
                                      tgt_code=hyp.code,
                                      tgt_actions=None,
                                      tgt_ast=None)
                hyp_examples.append(hyp_example)
                # hyp.tokenized_code = len(self.transition_system.tokenize_code(hyp.code))
                # hyp.code_token_count = len(hyp.code.split(' '))

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
                        feat_val = feat.get_feat_value(example, hyp,
                                                       hyp_id=hyp_id, all_hyps=hyps,
                                                       transition_system=self.transition_system)
                        hyp.rerank_feature_values[feat_name] = feat_val

    def get_rerank_score(self, hyp, param):
        raise NotImplementedError

    def _filter_hyps(self, decode_results, is_valid_hyp):
        for i in range(len(decode_results)):
            valid_hyps = []
            for hyp in decode_results[i]:
                if is_valid_hyp(hyp):
                    valid_hyps.append(hyp)

            decode_results[i] = valid_hyps

    def filter_hyps_and_initialize_features(self, examples, decode_results):
        if not hasattr(decode_results[0][0], 'rerank_feature_values'):
            print('initializing rerank features for hypotheses...', file=sys.stderr)

            def is_valid_hyp(hyp):
                try:
                    self.transition_system.tokenize_code(hyp.code)
                    if hyp.code:
                        return True
                except:
                    return False

                return False

            self._filter_hyps(decode_results, is_valid_hyp)

            self.initialize_rerank_features(examples, decode_results)

    def compute_rerank_performance(self, examples, decode_results, evaluator=CachedExactMatchEvaluator(),
                                   param=None, fast_mode=False, verbose=False):
        self.filter_hyps_and_initialize_features(examples, decode_results)

        if param is None:
            param = self.parameter

        sorted_decode_results = []
        for example, hyps in zip(examples, decode_results):
            if hyps:
                new_hyp_scores = [self.get_rerank_score(hyp, param=param) for hyp in hyps]
                best_hyp_idx = np.argmax(new_hyp_scores)
                best_hyp = hyps[best_hyp_idx]

                if fast_mode:
                    sorted_decode_results.append([best_hyp])
                else:
                    sorted_decode_results.append([hyps[i] for i in np.argsort(new_hyp_scores)[::-1]])
            else:
                sorted_decode_results.append([])

            if verbose:
                gold_standard_idx = [i for i, hyp in enumerate(hyps) if hyp.is_correct]
                if gold_standard_idx and gold_standard_idx[0] != best_hyp_idx:
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

        metric = evaluator.evaluate_dataset(examples, sorted_decode_results, fast_mode=fast_mode)

        return metric

    def train(self, examples, decode_results, initial_performance=0., metric='accuracy'):
        raise NotImplementedError

    @property
    def feature_num(self):
        return len(self.features)

    def __getattr__(self, item):
        if item in self.feat_map:
            return self.feat_map.get(item)
        raise ValueError

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        feature_names = []
        for feat in self.features:
            if isinstance(feat, nn.Module):
                feat.save(os.path.join(path + '.%s' % feat.feature_name))
            feature_names.append(feat.feature_name)

        params = {
            'parameter': self.parameter,
            'feature_names': feature_names,
            'transition_system': self.transition_system
        }

        torch.save(params, path)

    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        feature_names = params['feature_names']
        features = []
        for feat_name in feature_names:
            feat_cls = Registrable.registered_components[feat_name]
            if issubclass(feat_cls, Savable):
                feat_inst = feat_cls.load(model_path + '.%s' % feat_name, cuda=cuda)
                feat_inst.eval()
            else:
                feat_inst = feat_cls()
            features.append(feat_inst)

        reranker = cls(features, params['parameter'], params['transition_system'])

        return reranker


class LinearReranker(Reranker):
    """a reranker using linear features"""

    def get_rerank_score(self, hyp, param=None):
        if param is None:
            param = self.parameter

        score = sum(feat_weight * hyp.rerank_feature_values[feat_name] for feat_name, feat_weight in param.items())

        return score

    def generate_nbest_list(self, examples, decode_results, nbest_file_name, target_tokenizer, hyp_tokenizer):
        f_src = open(nbest_file_name + '.src', 'w')
        f_tgt = open(nbest_file_name + '.tgt', 'w')
        f_hyp = open(nbest_file_name + '.hyp', 'w')

        self.filter_hyps_and_initialize_features(examples, decode_results)

        for e_id, (example, hyp_list) in enumerate(zip(examples, decode_results)):
            f_src.write(' '.join(example.src_sent) + '\n')
            f_tgt.write(' '.join(target_tokenizer(example)) + '\n')

            if not hyp_list:
                hyp_str = "{e_id} ||| pass ||| 0. |||".format(e_id=e_id)
                for feat_name in self.feat_map:
                    hyp_str += ' {}=0.0'.format(feat_name)
                hyp_str += '\n'
                f_hyp.write(hyp_str)

                continue

            # new_hyp_scores = [self.get_rerank_score(hyp) for hyp in hyp_list]

            # hyp_ranks = np.argsort(new_hyp_scores)[::-1]

            # for i, hyp_id in enumerate(hyp_ranks):
            for hyp_id in range(len(hyp_list)):
                hyp = hyp_list[hyp_id]
                code_tokens = hyp_tokenizer(hyp)
                # reranker_score = new_hyp_scores[hyp_id]

                # code_token_count={code_token_count_feat.get_feat_value(example, hyp)}
                hyp_str = "{e_id} ||| {code_tokens} ||| {hyp_score} |||".format(e_id=e_id, code_tokens=' '.join(code_tokens), hyp_score=hyp.score)
                for feat_name in self.parameter.keys():
                    hyp_str += ' {feat_name}={feat_val}'.format(feat_name=feat_name, feat_val=hyp.rerank_feature_values[feat_name])
                hyp_str += '\n'
                f_hyp.write(hyp_str)

        f_src.close()
        f_tgt.close()
        f_hyp.close()

    @staticmethod
    def prepare_travatar_inputs_for_lambda_dcs_dataset(
        reconstructor_path, paraphrase_identifier_path,
        dev_set_path, test_set_path,
        dev_decode_results_path, test_decode_results_path,
        nbest_output_path, nbest_output_file_suffix = ''
    ):
        """
        Args:
            reconstructor_path: Path to the trained reconstruction model
            paraphrase_identifier_path: Path to the trained paraphrase identification model
            dev_decode_results_path: Path to the saved n-best predictions by the semantic parser using beam search
            test_decode_results_path: Same as above, for the test set,
            nbest_output_path: Path to output folder that will contain the generated n-best files for travatar training
            nbest_output_file_suffix: Add certain suffix to generated files (e.g., `{file_name}.seed0`)
        """
        # TODO: that's ugly but we need to avoid cyclic imports
        from asdl.lang.lambda_dcs.logical_form import get_canonical_order_of_logical_form, ast_to_logical_form
        from model.paraphrase import ParaphraseIdentificationModel
        from model.reconstruction_model import Reconstructor

        dev_set = Dataset.from_bin_file(dev_set_path)
        test_set = Dataset.from_bin_file(test_set_path)

        dev_decode_results = pickle.load(open(dev_decode_results_path, 'rb'))
        test_decode_results = pickle.load(open(test_decode_results_path, 'rb'))

        features = [
            ParserScore(),
            Reconstructor.load(reconstructor_path).eval(),
            ParaphraseIdentificationModel.load(paraphrase_identifier_path).eval(),
            NormalizedParserScore(),
            HypCodeTokensCount()
        ]

        reranker = LinearReranker(
            features=features,
            # the actual parameter weights listed here are not important, since we just need to create an
            # instance of `LinearReranker`. I used Graham's `travatar` for feature weight tuning instead of
            # the different `Reranker`'s (e.g., `GridSearchReranker`) implemented here,
            parameter=OrderedDict([
                    ('parser_score', 1.),
                    ('reconstructor', 0.),
                    ('paraphrase_identifier', 0.),
                    ('normalized_parser_score', 0.),
                    ('word_cnt', 0.),
            ]),
            transition_system=features[1].transition_system
        )

        # tokenize code while canonicalizing the order of sub-trees for conjunctions/disjunctions
        def _get_order(name):
            if name == 'flight':
                return -200
            elif name == 'from':
                return -199
            elif name == 'to':
                return -198

            return sum(ord(x) for x in name)

        def target_tokenizer(example):
            lf = ast_to_logical_form(example.tgt_ast)
            ordered_lf = get_canonical_order_of_logical_form(lf, _get_order=_get_order)

            return reranker.transition_system.tokenize_code(ordered_lf.to_string())

        def hyp_tokenizer(hyp):
            lf = ast_to_logical_form(hyp.tree)
            ordered_lf = get_canonical_order_of_logical_form(lf, _get_order=_get_order)

            return reranker.transition_system.tokenize_code(ordered_lf.to_string())

        os.makedirs(nbest_output_path, exist_ok=True)

        reranker.generate_nbest_list(
            dev_set.examples, dev_decode_results,
            os.path.join(nbest_output_path, 'dev' + nbest_output_file_suffix),
            target_tokenizer=target_tokenizer,
            hyp_tokenizer=hyp_tokenizer
        )

        reranker.generate_nbest_list(
            test_set.examples, test_decode_results,
            os.path.join(nbest_output_path, 'test' + nbest_output_file_suffix),
            target_tokenizer=target_tokenizer,
            hyp_tokenizer=hyp_tokenizer
        )

    def train(self, examples, decode_results, initial_performance=0., metric='accuracy'):
        raise NotImplementedError


class GridSearchReranker(Reranker):
    """Grid search reranker"""

    def get_rerank_score(self, hyp, param):
        feat_vals = np.array(list(hyp.rerank_feature_values.values()))
        score = hyp.score + np.dot(param, feat_vals)
        # score = np.dot(param, feat_vals)

        return score

    def train(self, examples, decode_results, evaluator=CachedExactMatchEvaluator(), initial_performance=0.):
        """optimize the ranker on a dataset using grid search"""
        best_score = initial_performance
        best_param = np.zeros(self.feature_num)

        param_space = (np.array(p) for p in itertools.combinations(np.arange(0, 3.01, 0.01), self.feature_num))

        for param in param_space:
            score = self.compute_rerank_performance(examples, decode_results, fast_mode=True, evaluator=evaluator, param=param)
            if score > best_score:
                print('New param=%s, score=%.4f' % (param, score), file=sys.stderr)
                best_param = param
                best_score = score

        self.parameter = best_param

    def train_multiprocess(self, examples, decode_results, evaluator=CachedExactMatchEvaluator(), initial_performance=0., num_workers=8):
        """optimize the ranker on a dataset using grid search"""
        best_score = initial_performance
        best_param = np.zeros(self.feature_num)

        self.initialize_rerank_features(examples, decode_results)

        print('generating parameter list', file=sys.stderr)
        param_space = [p for p in itertools.combinations(np.arange(0, 2.01, 0.01), self.feature_num)]
        print('generating parameter list done', file=sys.stderr)

        global _examples
        _examples = examples
        global _decode_results
        _decode_results = decode_results
        global _evaluator
        _evaluator = evaluator
        global _ranker
        _ranker = self

        def _norm(_param):
            return sum(p ** 2 for p in _param)

        with multiprocessing.Pool(processes=num_workers) as pool:
            # segment the parameter space
            segment_size = int(len(param_space) / num_workers / 5)
            param_space_segments = []
            ptr = 0
            while ptr < len(param_space):
                param_space_segments.append(param_space[ptr: ptr + segment_size])
                ptr += segment_size
            print('generated %d parameter segments' % len(param_space_segments), file=sys.stderr)

            results = pool.imap_unordered(_rank_segment_worker, param_space_segments)

            for param, score in results:
                if score > best_score or score == best_score and _norm(param) < _norm(best_param):
                    print('[Main] New param=%s, score=%.4f' % (param, score), file=sys.stderr)
                    best_param = param
                    best_score = score

        self.parameter = best_param


class XGBoostReranker(Reranker):
    def __init__(self, features, transition_system=None):
        super(XGBoostReranker, self).__init__(features, transition_system=transition_system)

        params = {'objective': 'rank:ndcg', 'learning_rate': .1,
                  'gamma': 5.0, 'min_child_weight': 0.1,
                  'max_depth': 4, 'n_estimators': 5}

        self.ranker = xgb.sklearn.XGBRanker(**params)

    def get_feature_matrix(self, decode_results, train=False):
        x, y, group = [], [], []

        for hyps in decode_results:
            if hyps:
                for hyp in hyps:
                    label = 1 if hyp.is_correct else 0
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

    def train(self, examples, decode_results, evaluator=CachedExactMatchEvaluator(), initial_performance=0.):
        self.initialize_rerank_features(examples, decode_results)

        train_x, train_y, group_train = self.get_feature_matrix(decode_results, train=True)
        self.ranker.fit(train_x, train_y, group_train)

        train_acc = self.compute_rerank_performance(examples, decode_results, fast_mode=True, evaluator=evaluator)
        print('Dev acc: %f' % train_acc, file=sys.stderr)
