# coding=utf-8

from cStringIO import StringIO
from collections import Iterable

from asdl import *
from asdl_ast import AbstractSyntaxTree, RealizedField


def parse_lambda_expr_helper(s, offset):
    if s[offset] != '(':
        name = ''
        while offset < len(s) and s[offset] != ' ':
            name += s[offset]
            offset += 1

        node = Node(name)
        return node, offset
    else:
        # it's a sub-tree
        offset += 2
        name = ''
        while s[offset] != ' ':
            name += s[offset]
            offset += 1

        node = Node(name)
        # extract its child nodes

        while True:
            if s[offset] != ' ':
                raise ValueError('malformed string: node should have either had a '
                                 'close paren or a space at position %d' % offset)

            offset += 1
            if s[offset] == ')':
                offset += 1
                return node, offset
            else:
                child_node, offset = parse_lambda_expr_helper(s, offset)

            node.add_child(child_node)


def parse_lambda_expr(s):
    return parse_lambda_expr_helper(s, 0)[0]


class Node(object):
    def __init__(self, name, children=None):
        self.name = name
        self.parent = None
        self.children = list()
        if children:
            if isinstance(children, Iterable):
                for child in children:
                    self.add_child(child)
            elif isinstance(children, Node):
                self.add_child(children)
            else: raise ValueError('Wrong type for child nodes')

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def __hash__(self):
        code = hash(self.name)

        for child in self.children:
            code = code * 37 + hash(child)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.name != other.name:
            return False

        if len(self.children) != len(other.children):
            return False

        if self.name == 'and' or self.name == 'or':
            return sorted(self.children, key=lambda x: x.name) == sorted(other.children, key=lambda x: x.name)
        else:
            return self.children == other.children

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'Node[%s, %d children]' % (self.name, len(self.children))

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def to_string(self, sb=None):
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()

        if self.is_leaf:
            sb.write(self.name)
        else:
            sb.write('( ')
            sb.write(self.name)

            for child in self.children:
                sb.write(' ')
                child.to_string(sb)

            sb.write(' )')

        if is_root:
            return sb.getvalue()


def logical_form_to_ast(grammar, lf_node):
    if lf_node.name == 'lambda':
        # expr -> Lambda(var variable, var_type type, expr body)
        prod = grammar[('expr', 'Lambda')]

        var_node = lf_node.children[0]
        var_field = RealizedField(prod['variable'], var_node.name)

        var_type_node = lf_node.children[1]
        var_type_field = RealizedField(prod['type'], var_type_node.name)

        body_node = lf_node.children[2]
        body_ast_node = logical_form_to_ast(grammar, body_node)  # of type expr
        body_field = RealizedField(prod['body'], body_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      [var_field, var_type_field, body_field])
    elif lf_node.name == 'argmax' or lf_node.name == 'argmin':
        # expr -> Argmax(var variable, expr domain, expr body)
        if lf_node.name == 'argmax':
            prod = grammar[('expr', 'Argmax')]
        else:
            prod = grammar[('expr', 'Argmin')]

        var_node = lf_node.children[0]
        var_field = RealizedField(prod['variable'], var_node.name)

        domain_node = lf_node.children[2]
        domain_ast_node = logical_form_to_ast(grammar, domain_node)
        domain_field = RealizedField(prod['domain'], domain_ast_node)

        body_node = lf_node.children[1]
        body_ast_node = logical_form_to_ast(grammar, body_node)
        body_field = RealizedField(prod['body'], body_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      [var_field, domain_field, body_field])
    elif lf_node.name == 'and' or lf_node.name == 'or':
        # expr -> And(expr* arguments) | Or(expr* arguments)
        if lf_node.name == 'and':
            prod = grammar[('expr', 'And')]
        else:
            prod = grammar[('expr', 'Or')]

        arg_ast_nodes = []
        for arg_node in lf_node.children:
            arg_ast_node = logical_form_to_ast(grammar, arg_node)
            arg_ast_nodes.append(arg_ast_node)

        ast_node = AbstractSyntaxTree(prod.constructor.name,
                                      RealizedField(prod['arguments'], arg_ast_nodes))
    elif lf_node.name == '>' or lf_node.name == '=' or lf_node.name == '<':
        # expr -> Compare(cmp_op op, expr left, expr right)
        prod = grammar[('expr', 'Compare')]
        op_name = 'GreaterThan' if lf_node.name == '>' else 'Equal' if lf_node.name == '=' else 'LessThan'
        op_field = RealizedField(prod['op'], AbstractSyntaxTree(grammar[('Compare', op_name)]))

        left_node = lf_node.children[0]
        left_ast_node = logical_form_to_ast(grammar, left_node)
        left_field = RealizedField(prod['left'], left_ast_node)

        right_node = lf_node.children[1]
        right_ast_node = logical_form_to_ast(grammar, right_node)
        right_field = RealizedField(prod['right'], right_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      [op_field, left_field, right_field])
    elif lf_node.name in ['flight', 'airline', 'from', 'to', 'day', 'month', 'arrival_time',
                          'nonstop', 'has_meal', 'round_trip']:
        # expr -> Apply(pred predicate, expr* arguments)
        prod = grammar[('expr', 'Apply')]
        arg_ast_nodes = []
        for arg_node in lf_node.children:
            arg_ast_node = logical_form_to_ast(grammar, arg_node)
            arg_ast_nodes.append(arg_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      RealizedField(prod['arguments'], arg_ast_nodes))
    elif lf_node.name.startswith('$'):
        prod = grammar[('expr', 'Variable')]
        ast_node = AbstractSyntaxTree(prod,
                                      RealizedField(prod['variable'], lf_node.name))
    elif ':cl' in lf_node.name or ':pd' in lf_node.name or lf_node.name in ['ci0', 'ci1', 'ti0', 'ti1', 'da0', 'da1', 'al0']:
        prod = grammar[('expr', 'Entity')]
        ast_node = AbstractSyntaxTree(prod,
                                      RealizedField(prod['entity'], lf_node.name))
    else:
        raise NotImplementedError

    return ast_node


if __name__ == '__main__':
    asdl_desc = """
    var, ent, num, var_type

    expr = Variable(var variable)
    | Entity(ent entity)
    | Number(num number)
    | Apply(pred predicate, expr* arguments)
    | Argmax(var variable, expr domain, expr body)
    | Argmin(var variable, expr domain, expr body)
    | Count(var variable, expr body)
    | Exists(var variable, expr body)
    | Lambda(var variable, var_type type, expr body)
    | Max(var variable, expr body)
    | Min(var variable, expr body)
    | Sum(var variable, expr domain, expr body)
    | The(var variable, expr body)
    | Not(expr argument)
    | And(expr* arguments)
    | Or(expr* arguments)
    | Compare(cmp_op op, expr left, expr right)

    cmp_op = GreaterThan | Equal | LessThan
    """

    grammar = ASDLGrammar.from_text(asdl_desc)
    # lf = parse_lambda_expr('( lambda $0 e ( and ( flight $0 ) ( airline $0 al0 ) ( from $0 ci0 ) ( to $0 ci1 ) ) )')
    lf = parse_lambda_expr('al0')
    ast_tree = logical_form_to_ast(grammar, lf)
    pass
