from __future__ import print_function
import six
import argparse
import sys
from flask import Flask, url_for, jsonify, render_template
import json

from components.standalone_parser import StandaloneParser

app = Flask(__name__)
parsers = dict()


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--config_file', type=str, required=True,
                            help='Config file that specifies model to load, see online doc for an example')
    arg_parser.add_argument('--port', type=int, required=False, default=8081)

    return arg_parser


@app.route("/")
def default():
    return render_template('default.html')


@app.route('/parse/<dataset>/<utterance>', methods=['GET'])
def parse(utterance, dataset):

    parser = parsers[dataset]

    if six.PY2:
        utterance = utterance.encode('utf-8', 'ignore')

    hypotheses = parser.parse(utterance, debug=True)

    responses = dict()
    responses['hypotheses'] = []

    for hyp_id, hyp in enumerate(hypotheses):
        print('------------------ Hypothesis %d ------------------' % hyp_id)
        print(hyp.code)
        print(hyp.tree.to_string())
        # print('Actions:')
        # for action_t in hyp.action_infos:
        #     print(action_t)

        actions_repr = [action.__repr__(True) for action in hyp.action_infos]

        hyp_entry = dict(id=hyp_id + 1,
                         value=hyp.code,
                         tree_repr=hyp.tree.to_string(),
                         score=hyp.score,
                         actions=actions_repr)

        responses['hypotheses'].append(hyp_entry)

    return jsonify(responses)


if __name__ == '__main__':
    args = init_arg_parser().parse_args()
    config_dict = json.load(open(args.config_file))

    for parser_id, config in config_dict.items():
        parser = StandaloneParser(parser_name=config['parser'],
                                  model_path=config['model_path'],
                                  example_processor_name=config['example_processor'],
                                  beam_size=config['beam_size'],
                                  cuda=args.cuda)

        parsers[parser_id] = parser

    app.run(host='0.0.0.0', port=args.port, debug=True)
