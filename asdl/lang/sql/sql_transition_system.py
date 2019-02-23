# coding=utf-8
from __future__ import absolute_import

import json

from common.registerable import Registrable
from ...asdl import ASDLGrammar
from ...asdl_ast import RealizedField, AbstractSyntaxTree
from ...transition_system import GenTokenAction, TransitionSystem, ApplyRuleAction, ReduceAction

from datasets.wikisql.lib.query import Query
from datasets.wikisql.lib.dbengine import DBEngine


cmp_op_idx2op_name = {0: 'Equal', 1: 'GreaterThan', 2: 'LessThan'}
ctr_name2cmp_op_idx = {v: k for k, v in cmp_op_idx2op_name.items()}
agg_idx2op_name = {1: 'Max', 2: 'Min', 3: 'Count', 4: 'Sum', 5: 'Avg'}
ctr_name2agg_idx = {v: k for k, v in agg_idx2op_name.items()}


class WikiSqlGenTokenAction(GenTokenAction):
    def __init__(self, token, copy_idx=-1):
        super(WikiSqlGenTokenAction, self).__init__(token)
        if not self.is_stop_signal() and copy_idx == -1:
            raise ValueError('token %s must be copied from source' % token)
        self.copy_idx = copy_idx

    @property
    def copy_from_src(self):
        return self.copy_idx >= 0 and not self.is_stop_signal()


class WikiSqlSelectColumnAction(GenTokenAction):
    def __init__(self, column_id):
        super(WikiSqlSelectColumnAction, self).__init__(column_id)

    @property
    def column_id(self):
        return self.token

    def __repr__(self):
        return 'SelectColumnAction[id=%s]' % self.column_id


def sql_query_to_asdl_ast(query, grammar):
    stmt_prod = grammar.get_prod_by_ctr_name('Select')

    # stmt = Select(agg_op? agg, column_name col_name, cond_expr* condition)

    if query.agg_index:
        agg_op_node = AbstractSyntaxTree(grammar.get_prod_by_ctr_name(agg_idx2op_name[query.agg_index]))
        agg_op_field = RealizedField(stmt_prod['agg'], agg_op_node)
    else:
        agg_op_field = RealizedField(stmt_prod['agg'])

    col_idx_field = RealizedField(stmt_prod['col_idx'], query.sel_index)

    condition_fields = RealizedField(stmt_prod['conditions'])

    for condition in query.conditions:
        col_idx, cmp_op_idx, val = condition
        cond_prod = grammar.get_prod_by_ctr_name('Condition')
        op_name = cmp_op_idx2op_name[cmp_op_idx]
        op_field = RealizedField(cond_prod['op'],
                                 AbstractSyntaxTree(grammar.get_prod_by_ctr_name(op_name)))
        cond_col_idx_field = RealizedField(cond_prod['col_idx'], col_idx)
        value_field = RealizedField(cond_prod['value'], val)

        condition_fields.add_value(AbstractSyntaxTree(cond_prod,
                                                      [op_field, cond_col_idx_field, value_field]))

    stmt_node = AbstractSyntaxTree(stmt_prod, [agg_op_field, col_idx_field, condition_fields])

    return stmt_node


def asdl_ast_to_sql_query(asdl_ast):
    # stmt = Select(agg_op? agg, column_name col_name, cond_expr* condition)
    sel_idx = asdl_ast['col_idx'].value
    agg_op_idx = 0 if asdl_ast['agg'].value is None else ctr_name2agg_idx[asdl_ast['agg'].value.production.constructor.name]
    conditions = []
    for condition_node in asdl_ast['conditions'].value:
        col_idx = condition_node['col_idx'].value
        cmp_op_idx = ctr_name2cmp_op_idx[condition_node['op'].value.production.constructor.name]
        value = condition_node['value'].value
        conditions.append((col_idx, cmp_op_idx, value))

    query = Query(sel_idx, agg_op_idx, conditions)

    return query


@Registrable.register('sql')
class SqlTransitionSystem(TransitionSystem):
    def ast_to_surface_code(self, asdl_ast):
        return asdl_ast_to_sql_query(asdl_ast)

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            elif hyp.frontier_field.type.name == 'column_idx':
                if hyp.frontier_field.cardinality == 'single':
                    return WikiSqlSelectColumnAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    return WikiSqlSelectColumnAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_primitive_field_actions(self, realized_field):
        if realized_field.type.name == 'column_idx' is not None:
            return [WikiSqlSelectColumnAction(int(realized_field.value))]
        elif realized_field.type.name == 'string':
            tokens = str(realized_field.value).split(' ') + ['</primitive>']
            return [GenTokenAction(token) for token in tokens]
        else:
            raise ValueError('unknown primitive field type')


def check():
    data_file = '/Users/yinpengcheng/Research/SemanticParsing/WikiSQL/data/train.jsonl'
    engine = DBEngine('/Users/yinpengcheng/Research/SemanticParsing/WikiSQL/data/train.db')
    grammar = ASDLGrammar.from_text(open('sql_asdl.txt').read())
    transition_system = SqlTransitionSystem(grammar)
    from asdl.hypothesis import Hypothesis
    for line in open(data_file):
        example = json.loads(line)
        query = Query.from_dict(example['sql'])
        asdl_ast = sql_query_to_asdl_ast(query, grammar)
        asdl_ast.sanity_check()
        actions = transition_system.get_actions(asdl_ast)
        hyp = Hypothesis()

        for action in actions:
            hyp.apply_action(action)

        # if asdl_ast_to_sql_query(hyp.tree) != asdl_ast_to_sql_query(asdl_ast):
        #     hyp_query = asdl_ast_to_sql_query(hyp.tree)
            # make sure the execution result is the same
            # hyp_query_result = engine.execute_query(example['table_id'], hyp_query)
            # ref_result = engine.execute_query(example['table_id'], query)

            # assert hyp_query_result == ref_result
        query_reconstr = asdl_ast_to_sql_query(asdl_ast)
        assert query == query_reconstr
        print(query)


if __name__ == '__main__':
    check()