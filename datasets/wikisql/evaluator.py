import sys
import traceback

from components.evaluator import Evaluator
from common.registerable import Registrable
from datasets.wikisql.lib.query import Query
from datasets.wikisql.lib.dbengine import DBEngine
from datasets.wikisql.utils import detokenize_query
from asdl.lang.sql.sql_transition_system import asdl_ast_to_sql_query


@Registrable.register('wikisql_evaluator')
class WikiSQLEvaluator(Evaluator):
    def __init__(self, transition_system, args):
        super(WikiSQLEvaluator, self).__init__(transition_system=transition_system)

        print(f'load evaluation database {args.sql_db_file}', file=sys.stderr)
        self.execution_engine = DBEngine(args.sql_db_file)
        self.answer_prune = args.answer_prune

    def is_hyp_correct(self, example, hyp):
        hyp_query = asdl_ast_to_sql_query(hyp.tree)
        detokenized_hyp_query = detokenize_query(hyp_query, example.meta, example.table)

        hyp_answer = self.execution_engine.execute_query(example.meta['table_id'], detokenized_hyp_query, lower=True)

        ref_query = Query.from_tokenized_dict(example.meta['query'])
        ref_answer = self.execution_engine.execute_query(example.meta['table_id'], ref_query, lower=True)

        result = ref_answer == hyp_answer

        return result

    def evaluate_dataset(self, examples, decode_results, fast_mode=False):
        if self.answer_prune:
            filtered_decode_results = []
            for example, hyp_list in zip(examples, decode_results):
                pruned_hyps = []
                if hyp_list:
                    for hyp_id, hyp in enumerate(hyp_list):
                        try:
                            # check if it is executable
                            detokenized_hyp_query = detokenize_query(hyp.code, example.meta, example.table)
                            hyp_answer = self.execution_engine.execute_query(example.meta['table_id'],
                                                                             detokenized_hyp_query, lower=True)
                            if len(hyp_answer) == 0:
                                continue

                            pruned_hyps.append(hyp)
                            if fast_mode: break
                        except:
                            print("Exception in converting tree to code:", file=sys.stdout)
                            print('-' * 60, file=sys.stdout)
                            print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                                      ' '.join(
                                                                                                          example.src_sent),
                                                                                                      example.tgt_code,
                                                                                                      hyp_id,
                                                                                                      hyp.tree.to_string()),
                                  file=sys.stdout)
                            print()
                            print(hyp.code)
                            traceback.print_exc(file=sys.stdout)
                            print('-' * 60, file=sys.stdout)

                filtered_decode_results.append(pruned_hyps)

            decode_results = filtered_decode_results

        eval_results = Evaluator.evaluate_dataset(self, examples, decode_results, fast_mode)

        return eval_results
