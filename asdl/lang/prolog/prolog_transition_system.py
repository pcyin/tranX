# coding=utf-8
from asdl.transition_system import TransitionSystem, GenTokenAction

try:
    from cStringIO import StringIO
except:
    from io import StringIO

from collections import Iterable
from asdl.asdl import *
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree

from common.registerable import Registrable


def prolog_node_to_ast(grammar, prolog_tokens, start_idx):
    node_name = prolog_tokens[start_idx]
    i = start_idx
    if node_name in ['job', 'language', 'loc', 'req_deg', 'application', 'area', 'company',
                     'des_deg', 'des_exp', 'platform', 'recruiter', 'req_exp', 'salary_greater_than',
                     'salary_less_than', 'title']:
        # it's a predicate
        prod = grammar.get_prod_by_ctr_name('Apply')
        pred_field = RealizedField(prod['predicate'], value=node_name)

        arg_ast_nodes = []
        i += 1
        assert prolog_tokens[i] == '('
        while True:
            i += 1
            arg_ast_node, end_idx = prolog_node_to_ast(grammar, prolog_tokens, i)
            arg_ast_nodes.append(arg_ast_node)

            i = end_idx
            if i >= len(prolog_tokens): break
            if prolog_tokens[i] == ')':
                i += 1
                break

            assert prolog_tokens[i] == ','

        arg_field = RealizedField(prod['arguments'], arg_ast_nodes)
        ast_node = AbstractSyntaxTree(prod, [pred_field, arg_field])
    elif node_name in ['ANS', 'X', 'A', 'B', 'P', 'J']:
        # it's a variable
        prod = grammar.get_prod_by_ctr_name('Variable')
        ast_node = AbstractSyntaxTree(prod,
                                      [RealizedField(prod['variable'], value=node_name)])

        i += 1
    elif node_name.endswith('id0') or node_name.endswith('id1') or node_name.endswith('id2') \
            or node_name in ['20', 'hour', 'num_salary', 'year', 'year0', 'year1', 'month']:
        # it's a literal
        prod = grammar.get_prod_by_ctr_name('Literal')
        ast_node = AbstractSyntaxTree(prod,
                                      [RealizedField(prod['literal'], value=node_name)])

        i += 1
    else:
        raise NotImplementedError

    return ast_node, i


def prolog_expr_to_ast_helper(grammar, prolog_tokens, start_idx=0):
    i = start_idx
    if prolog_tokens[i] == '(':
        i += 1

    parsed_nodes = []
    while True:
        if prolog_tokens[i] == '\\+':
            # expr -> Not(expr argument)
            prod = grammar.get_prod_by_ctr_name('Not')
            i += 1
            if prolog_tokens[i] == '(':
                arg_ast_node, end_idx = prolog_expr_to_ast_helper(grammar, prolog_tokens, i)
            else:
                arg_ast_node, end_idx = prolog_node_to_ast(grammar, prolog_tokens, i)
            i = end_idx

            assert arg_ast_node.production.type.name == 'expr'
            ast_node = AbstractSyntaxTree(prod,
                                          [RealizedField(prod['argument'], arg_ast_node)])

            parsed_nodes.append(ast_node)
        elif prolog_tokens[i] == '(':
            ast_node, end_idx = prolog_expr_to_ast_helper(grammar, prolog_tokens, i)
            parsed_nodes.append(ast_node)
            i = end_idx
        else:
            ast_node, end_idx = prolog_node_to_ast(grammar, prolog_tokens, i)
            parsed_nodes.append(ast_node)
            i = end_idx

        if i >= len(prolog_tokens): break
        if prolog_tokens[i] == ')':
            i += 1
            break

        if prolog_tokens[i] == ',':
            # and
            i += 1
        elif prolog_tokens[i] == ';':
            # Or
            prod = grammar.get_prod_by_ctr_name('Or')

            assert parsed_nodes
            if len(parsed_nodes) == 1:
                left_ast_node = parsed_nodes[0]
            else:
                left_expr_prod = grammar.get_prod_by_ctr_name('And')
                left_ast_node = AbstractSyntaxTree(left_expr_prod, [RealizedField(left_expr_prod['arguments'], parsed_nodes)])
                parsed_nodes = []

            # get the right ast node
            i += 1
            right_ast_node, end_idx = prolog_expr_to_ast_helper(grammar, prolog_tokens, i)

            ast_node = AbstractSyntaxTree(prod,
                                          [RealizedField(prod['left'], left_ast_node),
                                           RealizedField(prod['right'], right_ast_node)])

            i = end_idx
            parsed_nodes = [ast_node]

            if i >= len(prolog_tokens): break
            if prolog_tokens[i] == ')':
                i += 1
                break

    assert parsed_nodes
    if len(parsed_nodes) > 1:
        prod = grammar.get_prod_by_ctr_name('And')
        return_node = AbstractSyntaxTree(prod, [RealizedField(prod['arguments'], parsed_nodes)])
    else:
        return_node = parsed_nodes[0]

    return return_node, i


def prolog_expr_to_ast(grammar, prolog_expr):
    prolog_tokens = prolog_expr.strip().split(' ')
    return prolog_expr_to_ast_helper(grammar, prolog_tokens, start_idx=0)[0]


def ast_to_prolog_expr(asdl_ast):
    sb = StringIO()
    constructor_name = asdl_ast.production.constructor.name
    if constructor_name == 'Apply':
        predicate = asdl_ast['predicate'].value
        sb.write(predicate)
        sb.write(' (')
        for i, arg in enumerate(asdl_ast['arguments'].value):
            arg_val = arg.fields[0].value
            if i == 0: sb.write(' ')
            else: sb.write(' , ')
            sb.write(arg_val)
        
        sb.write(' )')
    elif constructor_name == 'And':
        for i, arg_ast in enumerate(asdl_ast['arguments'].value):
            arg_str = ast_to_prolog_expr(arg_ast)
            if i > 0: sb.write(' , ')
            if arg_ast.production.constructor.name == 'Or':
                sb.write('( ')
                sb.write(arg_str)
                sb.write(' )')
            else:
                sb.write(arg_str)
    elif constructor_name == 'Or':
        left_ast = asdl_ast['left'].value
        right_ast = asdl_ast['right'].value

        left_ast_str = ast_to_prolog_expr(left_ast)
        right_ast_str = ast_to_prolog_expr(right_ast)

        if left_ast.production.constructor.name == 'Apply':
            sb.write('( ')
            sb.write(left_ast_str)
            sb.write(' )')
        else:
            sb.write(left_ast_str)

        sb.write(' ; ')

        if right_ast.production.constructor.name in ('Apply', 'And'):
            sb.write('( ')
            sb.write(right_ast_str)
            sb.write(' )')
        else:
            sb.write(right_ast_str)
    elif constructor_name == 'Not':
        sb.write('\\+ ')
        arg_ast = asdl_ast['argument'].value
        arg_str = ast_to_prolog_expr(arg_ast)
        if arg_ast.production.constructor.name in ('Or', 'And'):
            sb.write('( ')
            sb.write(arg_str)
            sb.write(' )')
        else:
            sb.write(arg_str)

    return sb.getvalue()


def is_equal_ast(this_ast, other_ast):
    if not isinstance(other_ast, this_ast.__class__):
        return False

    if this_ast == other_ast:
        return True

    if isinstance(this_ast, AbstractSyntaxTree):
        if this_ast.production != other_ast.production:
            return False

        if len(this_ast.fields) != len(other_ast.fields):
            return False

        for i in range(len(this_ast.fields)):
            if this_ast.production.constructor.name in ('And', 'Or') and this_ast.fields[i].name == 'arguments':
                this_field_val = sorted(this_ast.fields[i].value, key=lambda x: x.to_string())
                other_field_val = sorted(other_ast.fields[i].value, key=lambda x: x.to_string())
            else:
                this_field_val = this_ast.fields[i].value
                other_field_val = other_ast.fields[i].value

            if not is_equal_ast(this_field_val, other_field_val): return False
    elif isinstance(this_ast, list):
        if len(this_ast) != len(other_ast): return False

        for i in range(len(this_ast)):
            if not is_equal_ast(this_ast[i], other_ast[i]): return False
    else:
        return this_ast == other_ast

    return True


@Registrable.register('prolog')
class PrologTransitionSystem(TransitionSystem):
    def compare_ast(self, hyp_ast, ref_ast):
        return is_equal_ast(hyp_ast, ref_ast)

    def ast_to_surface_code(self, asdl_ast):
        return ast_to_prolog_expr(asdl_ast)

    def surface_code_to_ast(self, code):
        return prolog_expr_to_ast(self.grammar, code)

    def hyp_correct(self, hyp, example):
        return is_equal_ast(hyp.tree, example.tgt_ast)

    def tokenize_code(self, code, mode):
        return code.split(' ')

    def get_primitive_field_actions(self, realized_field):
        assert realized_field.cardinality == 'single'
        if realized_field.value is not None:
            return [GenTokenAction(realized_field.value)]
        else:
            return []

if __name__ == '__main__':
    pass
