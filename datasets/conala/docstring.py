import json
import sys
import os
import argparse


def docstring2conala(args):
    if args.classmethod:
        snippet_file = os.path.join(args.inp, 'parallel_methods_decl')
        intent_file = os.path.join(args.inp, 'parallel_methods_desc')
        meta_file = os.path.join(args.inp, 'parallel_methods_meta')
    else:
        snippet_file = os.path.join(args.inp, 'parallel_decl')
        intent_file = os.path.join(args.inp, 'parallel_desc')
        meta_file = os.path.join(args.inp, 'parallel_meta')

    snippet_file = open(snippet_file, 'r', encoding='latin-1')
    intent_file = open(intent_file, 'r', encoding='latin-1')
    meta_file = open(meta_file, 'r', encoding='latin-1')

    num_pairs = 0

    with open(args.out, 'w') as fout:
        for i, snippet in enumerate(snippet_file):
            snippet = snippet.strip()
            if not snippet.startswith('def '):
                continue
            snippet = snippet[4:-1]

            intent = intent_file.readline().strip()[1:-1]
            intent = intent.split('DCNL', 1)[0].strip()

            meta = meta_file.readline().strip()

            if len(intent) <= 0 or len(snippet) <= 0:
                continue

            num_pairs += 1

            result = {
                'snippet': snippet,
                'intent': intent,
                'question_id': i + 1,
                'from': meta
            }

            fout.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inp', type=str, help='Path to v2 parallel dir')
    arg_parser.add_argument('--out', type=str, help='Output file')
    arg_parser.add_argument('--classmethod', action='store_true')
    args = arg_parser.parse_args()

    docstring2conala(args)
