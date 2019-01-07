from __future__ import print_function
import sys
from flask import Flask, url_for, jsonify, render_template
import json

from components.standalone_parser import StandaloneParser

app = Flask(__name__)


parsers = {
    # 'atis': StandaloneParser('atis',
    #                          'saved_models/atis/'
    #                          'model.atis.sup.lstm.hidden200.embed128.action128.field32.type32.dropout0.3.lr_decay0.5.beam5.vocab.bin.train.bin.glorot.par_state_w_field_embed.seed0.bin',
    #                          beam_size=5,
    #                          cuda=False),
    # 'geo': StandaloneParser('geo',
    #                         'saved_models/geo/'
    #                         'model.geo.sup.lstm.hid256.embed128.act128.field32.type32.drop0.5.lr_decay0.985.lr_dec_aft20.beam5.vocab.freq2.bin.train.bin.pat1000.max_ep200.batch10.lr0.002.glorot.no_par_info.seed2.cu90.bin',
    #                         beam_size=5,
    #                         cuda=False),
    # 'django': StandaloneParser('django',
    #                            'saved_models/django/'
    #                            'model.sup.django.lstm.hidden256.embed128.action128.field64.type64.dropout0.3.lr0.001.lr_decay0.5.beam_size15.vocab.0513.freq5.bin.train.0513.bin.glorot.par_state_w_field_embe.seed0.bin',
    #                            beam_size=15, cuda=False),
    'conala': StandaloneParser('conala',
                              'saved_models/conala/'
                              'model.sup.conala.lstm.hidden256.embed128.action128.field64.type64.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.vocab.var_str_sep.new_dev.src_freq3.code_freq3.bin.train.var_str_sep.new_dev.bin.glorot.par_state.seed1.bin',
                              beam_size=15,
                              cuda=False)
}


@app.route("/")
def default():
    return render_template('default.html')


@app.route('/parse/<dataset>/<utterance>', methods=['GET'])
def parse(utterance, dataset='atis'):

    parser = parsers[dataset]

    if sys.version_info < (3, 0):
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
    app.run(host='0.0.0.0', port=8081, debug=True)
