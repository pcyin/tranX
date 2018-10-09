## ASDL Transition System

This package contains a standalone transition system based on the ASDL formalism, 
and its instantiations in different languages (lambda calculus, prolog, Python, SQL).
A transition system defines the set of tree-constructing actions to generate an ASDL AST.
This package can be used as a standalone component independent of `tranX`.

### File Structure

* `asdl.py` contains classes that implement basic concepts in ASDL (grammar, constructor, production, field, type, etc.)
* `asdl_ast.py` contains the `AbstractSyntaxTree` class that define an abstract syntax tree
* `transition_system.py` contains the abstract class of a transition system,
instantiated by the transition systems in each language folder. 
A transition system defines the set of tree-constructing actions used to generate an AST. 
* `hypothesis.py` contains the `Hypothesis` class, which records the state of a partially generated AST constructed
by a series of actions. 

### Example

Below is an example usage of the `PythonTransitionSystem`, which defines the actions
to generate Python code snippets.

```python
# coding=utf-8

import ast
from asdl.asdl import ASDLGrammar
from asdl.lang.py.py_asdl_helper import *
from asdl.lang.py.py_transition_system import *
from asdl.hypothesis import *
import astor

# read in the grammar specification of Python 2.7, defined in ASDL
asdl_text = open('py_asdl.txt').read()
grammar = ASDLGrammar.from_text(asdl_text)

py_code = """pandas.read('file.csv', nrows=100)"""

# get the (domain-specific) python AST of the example Python code snippet
py_ast = ast.parse(py_code)

# convert the python AST into general-purpose ASDL AST used by tranX
asdl_ast = python_ast_to_asdl_ast(py_ast.body[0], grammar)
print('String representation of the ASDL AST: \n%s' % asdl_ast.to_string())
print('Size of the AST: %d' % asdl_ast.size)

# we can also convert the ASDL AST back into Python AST
py_ast_reconstructed = asdl_ast_to_python_ast(asdl_ast, grammar)

# initialize the Python transition parser
parser = PythonTransitionSystem(grammar)

# get the sequence of gold-standard actions to construct the ASDL AST
actions = parser.get_actions(asdl_ast)

# a hypothesis is an (partial) ASDL AST generated using a sequence of tree-construction actions
hypothesis = Hypothesis()
for t, action in enumerate(actions, 1):
    # the type of the action should belong to one of the valid continuing types
    # of the transition system
    assert action.__class__ in parser.get_valid_continuation_types(hypothesis)

    # if it's an ApplyRule action, the production rule should belong to the
    # set of rules with the same LHS type as the current rule
    if isinstance(action, ApplyRuleAction) and hypothesis.frontier_node:
        assert action.production in grammar[hypothesis.frontier_field.type]

    print('t=%d, Action=%s' % (t, action))
    hypothesis.apply_action(action)

# get the surface code snippets from the original Python AST,
# the reconstructed AST and the AST generated using actions
# they should be the same
src1 = astor.to_source(py_ast).strip()
src2 = astor.to_source(py_ast_reconstructed).strip()
src3 = astor.to_source(asdl_ast_to_python_ast(hypothesis.tree, grammar)).strip()

assert src1 == src2 == src3 == "pandas.read('file.csv', nrows=100)"

```