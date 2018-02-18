# coding=utf-8

import ast

from asdl.asdl import ASDLGrammar
from asdl.lang.py.py_asdl_helper import *
from asdl.lang.py.py_transition_system import *

if __name__ == '__main__':
    asdl_text = open('py_asdl.txt').read()
    grammar = ASDLGrammar.from_text(asdl_text)
    py_code = 'sorted(mylist, reverse=True)'
    #py_code = 'a = dict({a: None, b:False, s:"I love my mother", sd:124+3})'
    #py_code = '1e10'
    py_ast = ast.parse(py_code)
    asdl_ast = python_ast_to_asdl_ast(py_ast.body[0], grammar)
    print(asdl_ast.to_string())
    print(asdl_ast.size)
    py_ast_reconstructed = asdl_ast_to_python_ast(asdl_ast, grammar)

    asdl_ast2 = asdl_ast.copy()
    assert asdl_ast == asdl_ast2
    del asdl_ast2

    parser = PythonTransitionSystem(grammar)
    actions = parser.get_actions(asdl_ast)

    from asdl.hypothesis import *
    hyp = Hypothesis()
    for action in actions:
        # assert action.__class__ in parser.get_valid_continuation_types(hyp)
        # if isinstance(action, ApplyRuleAction):
        #     assert action.production in grammar[hyp.frontier_field.type]
        print(action)
        hyp.apply_action(action)

    import astor
    src1 = astor.to_source(py_ast)
    src2 = astor.to_source(py_ast_reconstructed)
    src3 = astor.to_source(asdl_ast_to_python_ast(hyp.tree, grammar))

    print(src3)
    assert src1 == src3
    pass
