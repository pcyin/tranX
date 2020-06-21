from datasets.wikisql.lib.dbengine import DBEngine
import json

from components.evaluator import Evaluator


class WikiSqlEvaluator(Evaluator):
    def __init__(self, args):
        super(WikiSqlEvaluator, self).__init__()
        self.db_engine = DBEngine(args.wikisql_db_file)
        self.example_dicts = []
        for json_line in open(args.wikisql_table_file):
            self.example_dicts.append(json.loads(json_line))
