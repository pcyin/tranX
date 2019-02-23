# coding=utf-8

from asdl.transition_system import TransitionSystem, GenTokenAction, ReduceAction

from .logical_form import ast_to_logical_form, logical_form_to_ast, Node, parse_lambda_expr

from common.registerable import Registrable


@Registrable.register('lambda_dcs')
class LambdaCalculusTransitionSystem(TransitionSystem):
    def tokenize_code(self, code, mode=None):
        return code.strip().split(' ')

    def surface_code_to_ast(self, code):
        return logical_form_to_ast(self.grammar, parse_lambda_expr(code))

    def compare_ast(self, hyp_ast, ref_ast):
        ref_lf = ast_to_logical_form(ref_ast)
        hyp_lf = ast_to_logical_form(hyp_ast)

        return ref_lf == hyp_lf

    def ast_to_surface_code(self, asdl_ast):
        lf = ast_to_logical_form(asdl_ast)
        code = lf.to_string()

        return code

    def get_primitive_field_actions(self, realized_field):
        assert realized_field.cardinality == 'single'
        if realized_field.value is not None:
            return [GenTokenAction(realized_field.value)]
        else:
            return []

    def is_valid_hypothesis(self, hyp, **kwargs):
        return True
