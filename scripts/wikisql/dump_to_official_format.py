# coding=utf-8

from datasets.wikisql.utils import detokenize_query
import json, argparse
import pickle


def dump_wiki_sql_eval_file(dataset, decode_results, output_file):
    f = open(output_file, 'w')
    for example, hyps in zip(dataset, decode_results):
        result_dict = dict()
        if hyps:
            hyp = hyps[0]
            result_dict['error'] = False
            result_dict['query'] = detokenize_query(hyp.code, example.meta, example.table).to_dict()
        else:
            result_dict['error'] = True

        json_line = json.dumps(result_dict)
        f.write(json_line + '\n')
    f.close()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset', default='data/wikisql/test.bin')
arg_parser.add_argument('--decode', required=True)
arg_parser.add_argument('--output', required=True)

args = arg_parser.parse_args()
decode_results = pickle.load(open(args.decode, 'rb'))
dataset = pickle.load(open(args.dataset, 'rb'))
dump_wiki_sql_eval_file(dataset, decode_results, args.output)
