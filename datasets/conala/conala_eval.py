#!/usr/bin/env python
import json
import sys
import os
import os.path
import re
import token
import tokenize
import argparse

import datasets.conala.bleu_score as bleu_score

# Main function for CodaLab evaluation purposes
def main():

    p = argparse.ArgumentParser(description="Evaluator for CoNaLa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--input_dir",
                   help="input directory, containing 'res/answer.txt' and 'ref/truth.txt'",
                   default=None)
    p.add_argument("--input_ref",
                   help="input reference file",
                   default=None)
    p.add_argument("--input_hyp",
                   help="input hypothesis file",
                   default=None)
    p.add_argument("--output_file",
                   help="output score file",
                   default=None)
    p.add_argument("--output_dir",
                   help="output score directory which will contain output_dir/scores.txt",
                   default=None)
    p.add_argument("--no_exact_match",
                   help="only output bleu scores and not exact_match score",
                   action="store_true")
    p.add_argument("--strip_ref_metadata",
                   help="strip metadata from the reference and get only the code",
                   action="store_true")

    args = p.parse_args()

    if not (args.input_dir or (args.input_ref and args.input_hyp)):
        raise ValueError("Must specify input_dir or input_ref+input_hyp")

    input_hyp = args.input_hyp if args.input_hyp else os.path.join(args.input_dir, 'res', 'answer.txt')
    input_ref = args.input_ref if args.input_ref else os.path.join(args.input_dir, 'ref', 'truth.txt')

    with open(input_hyp, 'r') as f_hyp:
        c_hyp = json.load(f_hyp)
        c_hyp = [tokenize_for_bleu_eval(s) for s in c_hyp]
    with open(input_ref, 'r') as f_ref:
        c_ref = json.load(f_ref)
        if args.strip_ref_metadata:
          c_ref = [x['snippet'] for x in c_ref]
        c_ref = [tokenize_for_bleu_eval(s) for s in c_ref]

    if len(c_hyp) != len(c_ref):
        raise ValueError('Length of hypothesis and reference don\'t match: {} != {}'.format(len(c_hyp), len(c_ref)))

    if args.output_file:
        f_out = open(args.output_file, 'w')
    elif args.output_dir:
        f_out = open(os.path.join(args.output_dir, 'scores.txt'), 'w')
    else:
        f_out = sys.stdout

    bleu_tup = bleu_score.compute_bleu([[x] for x in c_ref], c_hyp, smooth=False)
    bleu = bleu_tup[0]
    exact = sum([1 if h == r else 0 for h, r in zip(c_hyp, c_ref)])/len(c_hyp)

    f_out.write('bleu:{0:.2f}\n'.format(bleu * 100))
    if not args.no_exact_match:
        f_out.write('exact:{0:.2f}\n'.format(exact * 100))

    f_out.close()

""" Parses a file in the natural .jsonl format that the Conala corpus comes in.
    @param f: .jsonl file containing snippets
    @return: list of lists of tokens
"""
def parse_file_json(f):
    snippet_list = json.load(f)
    result = []
    for snippet in snippet_list:
        toks = tokenize_for_bleu_eval(snippet['snippet'])
        result.append(toks)
    return result

""" The tokenizer that we use for code submissions, from Wang Ling et al., Latent Predictor Networks for Code Generation (2016)
    @param code: string containing a code snippet
    @return: list of code tokens
"""
def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens

""" This runs the built-in Python tokenizer. Note that it only works on correctly parseable Python programs.
    @param string: string containing a Python tokenizable code snippet
    @return: list of code tokens
"""
def tokenize_code(string, concat_symbol=None):
    tokens = []
    string = string.strip().decode('utf-8').encode('ascii', 'strict') #.decode('string_escape')
    for toknum, tokval, _, _, _  in tokenize.generate_tokens(StringIO(string).readline):
        # We ignore these tokens during evaluation.
        if toknum not in [token.ENDMARKER, token.INDENT, token.DEDENT]:
            tokens.append(tokval.lower())

    return tokens

""" This builds the reference list for BLEU scoring
    @param reference_file_name: The reference file can be downloaded from https://conala-corpus.github.io/ as
                                conala_annotations.v1.0.zip/examples.annotated.test.json
    @return: list of references ready for BLEU scoring
"""
# 
def get_reference_list(reference_file_name):
    f_reference = open(reference_file_name)
    a = parse_file_json(f_reference)
    a = [[l] for l in a]
    return a

""" This scores hypotheses against references using BLEU.
    @param reference_list: reference list returned by get_reference_list.
    @param hypothesis_list: list of lists of tokens that a model generates.
    @return: 3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
             precisions and brevity penalty.
"""
def evaluate_bleu(reference_list, hypothesis_list):
    b = [tokenize_for_bleu_eval(s) for s in hypothesis_list]
    return bleu_score.compute_bleu(reference_list, b, smooth=False)

if __name__ == '__main__':
    main()
