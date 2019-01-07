from __future__ import print_function
import argparse
import sys

import torch

from model.utils import get_parser_class
from datasets.utils import get_example_processor_cls


def init_args():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    
    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_time_step', default=100, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')

    args = arg_parser.parse_args()

    return args


class StandaloneParser(object):
    """
    a tranX parser that could parse raw input issued by end user, it is a
    bundle of a `Parser` and an `ExampleProcessor`. It is useful for demo
    purposes
    """

    def __init__(self, dataset_name, model_path, beam_size=5, cuda=False):
        parser_saved_args = torch.load(model_path, 
                                       map_location=lambda storage, loc: storage)['args']
        print('load parser from [%s]' % model_path, file=sys.stderr)
        parser = get_parser_class(parser_saved_args.lang).load(model_path, cuda=cuda)
        parser.eval()

        self.parser = parser
        self.example_processor = get_example_processor_cls(dataset_name)(parser.transition_system)
        self.beam_size = beam_size

    def parse(self, utterance, debug=False):
        utterance = utterance.strip()
        processed_utterance_tokens, utterance_meta = self.example_processor.pre_process_utterance(utterance)
        print(processed_utterance_tokens)
        hypotheses = self.parser.parse(processed_utterance_tokens, beam_size=self.beam_size, debug=debug)

        valid_hypotheses = list(filter(lambda hyp: self.parser.transition_system.is_valid_hypothesis(hyp), hypotheses))

        for hyp in valid_hypotheses:
            self.example_processor.post_process_hypothesis(hyp, utterance_meta)

        # for hyp_id, hyp in enumerate(valid_hypotheses):
        #     print('------------------ Hypothesis %d ------------------' % hyp_id)
        #     print(hyp.code)
        #     print(hyp.tree.to_string())
        #     print('Actions:')
        #     for action_t in hyp.action_infos:
        #         print(action_t.action)

        return valid_hypotheses
