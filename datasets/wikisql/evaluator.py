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

        self.execution_engine = DBEngine(args.sql_db_file)

    def is_hyp_correct(self, example, hyp):
        hyp_query = asdl_ast_to_sql_query(hyp.tree)
        ref_query = Query.from_tokenized_dict(example.meta['query'])
        detokenized_hyp_query = detokenize_query(hyp_query, example.meta, example.table)

        # result = detokenized_hyp_query == ref_query
        ref_answer = self.execution_engine.execute_query(example.meta['table_id'], ref_query, lower=True)
        hyp_answer = self.execution_engine.execute_query(example.meta['table_id'], detokenized_hyp_query, lower=True)

        result = ref_answer == hyp_answer

        return result
