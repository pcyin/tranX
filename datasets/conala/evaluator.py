from common.evaluator import Evaluator
from common.registerable import Registrable
from components.dataset import Dataset
from .util import decanonicalize_code
from .conala_eval import tokenize_for_bleu_eval
from .bleu_score import compute_bleu
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
import ast
import astor


@Registrable.register('conala_evaluator')
class ConalaEvaluator(Evaluator):
    def __init__(self, transition_system=None):
        super(ConalaEvaluator, self).__init__()
        self.transition_system = transition_system
        self.default_metric = 'corpus_bleu'

    def is_hyp_correct(self, example, hyp):
        ref_code = example.tgt_code
        ref_py_ast = ast.parse(ref_code)
        ref_reformatted_code = astor.to_source(ref_py_ast).strip()

        ref_code_tokens = self.transition_system.tokenize_code(ref_reformatted_code)
        hyp_code_tokens = self.transition_system.tokenize_code(hyp.code)

        return ref_code_tokens == hyp_code_tokens

    def get_sentence_bleu(self, example, hyp):
        return sentence_bleu([tokenize_for_bleu_eval(example.meta['example_dict']['snippet'])],
                             tokenize_for_bleu_eval(hyp.decanonical_code),
                             smoothing_function=SmoothingFunction().method3)

    def evaluate_dataset(self, dataset, decode_results, fast_mode=False):
        examples = dataset.examples if isinstance(dataset, Dataset) else dataset
        assert len(examples) == len(decode_results)

        # speed up, cache tokenization results
        if not hasattr(examples[0], 'reference_code_tokens'):
            for example in examples:
                setattr(example, 'reference_code_tokens', tokenize_for_bleu_eval(example.meta['example_dict']['snippet']))

        if not hasattr(decode_results[0][0], 'decanonical_code_tokens'):
            for i, example in enumerate(examples):
                hyp_list = decode_results[i]
                # here we prune any hypothesis that throws an error when converting back to the decanonical code!
                # This modifies the decode_results in-place!
                filtered_hyp_list = []
                for hyp in hyp_list:
                    if not hasattr(hyp, 'decanonical_code'):
                        try:
                            hyp.decanonical_code = decanonicalize_code(hyp.code, slot_map=example.meta['slot_map'])
                            if hyp.decanonical_code:
                                hyp.decanonical_code_tokens = tokenize_for_bleu_eval(hyp.decanonical_code)
                                filtered_hyp_list.append(hyp)
                        except: pass

                decode_results[i] = filtered_hyp_list

        if fast_mode:
            references = [e.reference_code_tokens for e in examples]
            hypotheses = [hyp_list[0].decanonical_code_tokens if hyp_list else [] for hyp_list in decode_results]

            bleu_tup = compute_bleu([[x] for x in references], hypotheses, smooth=False)
            bleu = bleu_tup[0]

            return bleu
        else:
            tokenized_ref_snippets = []
            hyp_code_tokens = []
            best_hyp_code_tokens = []
            sm_func = SmoothingFunction().method3
            sent_bleu_scores = []
            oracle_bleu_scores = []
            oracle_exact_match = []
            for example, hyp_list in zip(examples, decode_results):
                tokenized_ref_snippets.append(example.reference_code_tokens)
                example_hyp_bleu_scores = []
                if hyp_list:
                    for i, hyp in enumerate(hyp_list):
                        hyp.bleu_score = sentence_bleu([example.reference_code_tokens],
                                                       hyp.decanonical_code_tokens,
                                                       smoothing_function=sm_func)
                        hyp.is_correct = self.is_hyp_correct(example, hyp)

                        example_hyp_bleu_scores.append(hyp.bleu_score)

                    top_decanonical_code_tokens = hyp_list[0].decanonical_code_tokens
                    sent_bleu_score = hyp_list[0].bleu_score

                    best_hyp_idx = np.argmax(example_hyp_bleu_scores)
                    oracle_sent_bleu = example_hyp_bleu_scores[best_hyp_idx]
                    _best_hyp_code_tokens = hyp_list[best_hyp_idx].decanonical_code_tokens
                else:
                    top_decanonical_code_tokens = []
                    sent_bleu_score = 0.
                    oracle_sent_bleu = 0.
                    _best_hyp_code_tokens = []

                oracle_exact_match.append(any(hyp.is_correct for hyp in hyp_list))
                hyp_code_tokens.append(top_decanonical_code_tokens)
                sent_bleu_scores.append(sent_bleu_score)
                oracle_bleu_scores.append(oracle_sent_bleu)
                best_hyp_code_tokens.append(_best_hyp_code_tokens)

            bleu_tup = compute_bleu([[x] for x in tokenized_ref_snippets], hyp_code_tokens, smooth=False)
            corpus_bleu = bleu_tup[0]

            bleu_tup = compute_bleu([[x] for x in tokenized_ref_snippets], best_hyp_code_tokens, smooth=False)
            oracle_corpus_bleu = bleu_tup[0]

            avg_sent_bleu = np.average(sent_bleu_scores)
            oracle_avg_sent_bleu = np.average(oracle_bleu_scores)
            exact = sum([1 if h == r else 0 for h, r in zip(hyp_code_tokens, tokenized_ref_snippets)]) / float(
                len(examples))
            oracle_exact_match = np.average(oracle_exact_match)

            return {'corpus_bleu': corpus_bleu,
                    'oracle_corpus_bleu': oracle_corpus_bleu,
                    'avg_sent_bleu': avg_sent_bleu,
                    'oracle_avg_sent_bleu': oracle_avg_sent_bleu,
                    'exact_match': exact,
                    'oracle_exact_match': oracle_exact_match}
