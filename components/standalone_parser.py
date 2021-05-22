from __future__ import print_function

import sys

import six

from common.registerable import Registrable
from components.reranker import GridSearchReranker
from components.dataset import Example
from model.parser import Parser
from model.reconstruction_model import Reconstructor
from model.paraphrase import ParaphraseIdentificationModel
from datasets.conala import example_processor
from datasets.atis import example_processor
from datasets.geo import example_processor
from datasets.django import example_processor

if six.PY3:
    pass


class StandaloneParser(object):
    """
    a tranX parser that could parse raw input issued by end user, it is a
    bundle of a `Parser` and an `ExampleProcessor`. It is useful for demo
    purposes
    """

    def __init__(self, parser_name, model_path, example_processor_name, beam_size=5, reranker_path=None, cuda=False):
        print('load parser from [%s]' % model_path, file=sys.stderr)

        self.parser = parser = Registrable.by_name(parser_name).load(model_path, cuda=cuda).eval()
        self.reranker = None
        if reranker_path:
            self.reranker = GridSearchReranker.load(reranker_path)
        self.example_processor = Registrable.by_name(example_processor_name)(parser.transition_system)
        self.beam_size = beam_size

    def parse(self, utterance, debug=False):
        utterance = utterance.strip()
        processed_utterance_tokens, utterance_meta = self.example_processor.pre_process_utterance(utterance)
        print(processed_utterance_tokens)
        print(utterance_meta)
        examples = [Example(idx=None,
                          src_sent=processed_utterance_tokens,
                          tgt_code=None,
                          tgt_actions=None,
                          tgt_ast=None)]
        hypotheses = self.parser.parse(processed_utterance_tokens, beam_size=self.beam_size, debug=debug)

        if self.reranker:
            hypotheses = self.decode_tree_to_code(hypotheses)
            hypotheses = self.reranker.rerank_hypotheses(examples, [hypotheses])[0]

        valid_hypotheses = list(filter(lambda hyp: self.parser.transition_system.is_valid_hypothesis(hyp), hypotheses))
        for hyp in valid_hypotheses:
            self.example_processor.post_process_hypothesis(hyp, utterance_meta)

        for hyp_id, hyp in enumerate(valid_hypotheses):
            print('------------------ Hypothesis %d ------------------' % hyp_id)
            print(hyp.code)
            print(hyp.tree.to_string())
            print('Actions:')
            for action_t in hyp.action_infos:
                print(action_t.action)

        return valid_hypotheses

    def decode_tree_to_code(self, hyps):
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            try:
                hyp.code = self.parser.transition_system.ast_to_surface_code(hyp.tree)
                decoded_hyps.append(hyp)
            except:
                pass
        return decoded_hyps