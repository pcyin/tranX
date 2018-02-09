# coding=utf-8

from asdl.transition_system import TransitionSystem, GenTokenAction, ReduceAction
from asdl.asdl import *
from asdl.asdl_ast import *

from logical_form import ast_to_logical_form, logical_form_to_ast, Node


class LambdaCalculusTransitionSystem(TransitionSystem):
    def tokenize_code(self, code, mode=None):
        return code.strip().split(' ')

    def hyp_correct(self, hyp, example):
        ref_ast = example.tgt_ast
        ref_lf = ast_to_logical_form(ref_ast)
        hyp_lf = ast_to_logical_form(hyp.tree)

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
