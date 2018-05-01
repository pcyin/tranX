# coding=utf-8
from asdl.transition_system import GenTokenAction


LOGICAL_FORM_LEXICON = {
    'city:t': ['citi'],
    'density:i': ['densiti', 'averag', 'popul'],
}


class AttentionUtil(object):
    @staticmethod
    def get_candidate_tokens_to_attend(src_tokens, action):
        tokens_to_attend = dict()
        if isinstance(action, GenTokenAction):
            tgt_token = action.token
            for src_idx, src_token in enumerate(src_tokens):
                # match lemma
                if len(src_token) >= 3 and tgt_token.startswith(src_token) or \
                                src_token in LOGICAL_FORM_LEXICON.get(tgt_token, []):
                    tokens_to_attend[src_idx] = src_token

            # print(tokens_to_attend, tgt_token)
        return tokens_to_attend
