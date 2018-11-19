from .util import decanonicalize_code
from .conala_eval import tokenize_for_bleu_eval
from .bleu_score import compute_bleu
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np


def evaluate(examples=None, decode_results=None, references=None, hypotheses=None):
    if references and hypotheses:
        bleu_tup = compute_bleu([[x] for x in references], hypotheses, smooth=False)
        bleu = bleu_tup[0]

        return bleu
    else:
        ref_snippets = [e.meta['example_dict']['snippet'] for e in examples]
        hyp_code_list = []
        best_hyp_code_list = []
        sm_func = SmoothingFunction().method3
        sent_bleu_scores = []
        oracle_bleu_scores = []
        for example, hyp_list in zip(examples, decode_results):
            ref_code_tokens = tokenize_for_bleu_eval(example.meta['example_dict']['snippet'])
            example_hyp_bleu_scores = []
            if hyp_list:
                for i, hyp in enumerate(hyp_list):
                    hyp.bleu_score = sentence_bleu([ref_code_tokens], tokenize_for_bleu_eval(hyp.decanonical_code),
                                                   smoothing_function=sm_func)

                    if i == 0:
                        top_decanonical_code = hyp.decanonical_code
                        sent_bleu_scores.append(hyp.bleu_score)

                    example_hyp_bleu_scores.append(hyp.bleu_score)

                best_hyp_idx = np.argmax(example_hyp_bleu_scores)
                oracle_bleu = example_hyp_bleu_scores[best_hyp_idx]
                oracle_bleu_scores.append(oracle_bleu)
                best_hyp_code_list.append(hyp_list[best_hyp_idx].decanonical_code)
            else:
                top_decanonical_code = ''

            hyp_code_list.append(top_decanonical_code)

        c_hyp = [tokenize_for_bleu_eval(s) for s in hyp_code_list]
        c_best_hyp = [tokenize_for_bleu_eval(s) for s in best_hyp_code_list]
        c_ref = [tokenize_for_bleu_eval(s) for s in ref_snippets]

        bleu_tup = compute_bleu([[x] for x in c_ref], c_hyp, smooth=False)
        bleu = bleu_tup[0]

        bleu_tup = compute_bleu([[x] for x in c_ref], c_best_hyp, smooth=False)
        oracle_bleu = bleu_tup[0]

        sent_bleu = np.average(sent_bleu_scores)
        oracle_sent_bleu = np.average(oracle_bleu_scores)
        exact = sum([1 if h == r else 0 for h, r in zip(c_hyp, c_ref)]) / len(c_hyp)

        return {'bleu': bleu, 'oracle_bleu': oracle_bleu,
                'avg_sent_bleu': sent_bleu, 'avg_oracle_sent_bleu': oracle_sent_bleu,
                'exact_match': exact}
