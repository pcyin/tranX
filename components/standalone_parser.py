from __future__ import print_function
import argparse
import sys
import six
import torch
from model import parser

from common.registerable import Registrable

if six.PY3:
    from datasets.conala.example_processor import ConalaExampleProcessor


class StandaloneParser(object):
    """
    a tranX parser that could parse raw input issued by end user, it is a
    bundle of a `Parser` and an `ExampleProcessor`. It is useful for demo
    purposes
    """

    def __init__(self, parser_name, model_path, example_processor_name, beam_size=5, cuda=False):
        # lazy loading
        from datasets.geo.example_processor import GeoQueryExampleProcessor
        from datasets.atis.example_processor import ATISExampleProcessor
        from datasets.django.example_processor import DjangoExampleProcessor

        print('load parser from [%s]' % model_path, file=sys.stderr)

        self.parser = parser = Registrable.by_name(parser_name).load(model_path, cuda=cuda).eval()
        self.example_processor = Registrable.by_name(example_processor_name)(parser.transition_system)
        self.beam_size = beam_size

    def parse(self, utterance, debug=False):
        utterance = utterance.strip()
        processed_utterance_tokens, utterance_meta = self.example_processor.pre_process_utterance(utterance)
        print(processed_utterance_tokens)
        hypotheses = self.parser.parse(processed_utterance_tokens, beam_size=self.beam_size, debug=debug)

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
