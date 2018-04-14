from asdl.lang.sql.lib.dbengine import DBEngine
import json


class WikiSqlEvaluator(Evaluator):
    def __init__(self, args):
        self.db_engine = DBEngine(args.wikisql_db_file)
        self.example_dicts = []
        for json_line in open(args.wikisql_table_file):
            self.example_dicts.append(json.loads(json_line))
