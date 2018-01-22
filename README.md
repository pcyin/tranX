# asdl_code_generation
A general-purpose transition parser for generating source code defined in ASDL from natural language

## Usage (with Python)

Python's abstract grammar is defined in `asdl/lang/py/py_asdl.txt`.

```
# coding=utf-8

import ast
import astor
from asdl.asdl import ASDLGrammar
from asdl.lang.py.py_asdl_helper import *
from asdl.lang.py.py_transition_system import *
from asdl.hypothesis import *

if __name__ == '__main__':
    asdl_text = open('py_asdl.txt').read()
    grammar = ASDLGrammar.from_text(asdl_text)
    py_code = 'sorted(mydict, key=mydict.get, reverse=True)'
    py_ast = ast.parse(py_code)
    asdl_ast = python_ast_to_asdl_ast(py_ast.body[0], grammar)
    py_ast_reconstructed = asdl_ast_to_python_ast(asdl_ast, grammar)

    asdl_ast2 = asdl_ast.copy()
    assert asdl_ast == asdl_ast2
    del asdl_ast2

    parser = PythonTransitionSystem(grammar)
    actions = parser.get_actions(asdl_ast)

    hyp = Hypothesis()
    for action in actions:
        assert action.__class__ in parser.get_valid_continuation_types(hyp)
        if isinstance(action, ApplyRuleAction):
           assert action.production in grammar[hyp.frontier_field.type]
        hyp.apply_action(action)

    src1 = astor.to_source(py_ast)
    src2 = astor.to_source(py_ast_reconstructed)
    src3 = astor.to_source(asdl_ast_to_python_ast(hyp.tree, grammar))

    assert src1 == src3
```
