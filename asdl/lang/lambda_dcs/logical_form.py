# coding=utf-8
import copy

try:
    from cStringIO import StringIO
except:
    from io import StringIO

from collections import Iterable

from asdl.asdl import *
from asdl.asdl_ast import AbstractSyntaxTree, RealizedField


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


def get_canonical_order_of_logical_form(lf, order_by='alphabet', _get_order=None):
    lf_copy = copy.deepcopy(lf)

    if _get_order is None:
        def _get_order(name):
            if name == 'flight':
                return -200
            elif name == 'from':
                return -199
            elif name == 'to':
                return -198

            return name

    def _order(_lf):
        if _lf.name in ('and', 'or'):
            _lf.children = sorted(_lf.children, key=lambda x: _get_order(x.name))
        for child in _lf.children:
            _order(child)

    _order(lf_copy)

    return lf_copy


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
            else:
                raise ValueError('Wrong type for child nodes')

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
        prod = grammar.get_prod_by_ctr_name('Lambda')

        var_node = lf_node.children[0]
        var_field = RealizedField(prod['variable'], var_node.name)

        var_type_node = lf_node.children[1]
        var_type_field = RealizedField(prod['type'], var_type_node.name)

        body_node = lf_node.children[2]
        body_ast_node = logical_form_to_ast(grammar, body_node)  # of type expr
        body_field = RealizedField(prod['body'], body_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      [var_field, var_type_field, body_field])
    elif lf_node.name == 'argmax' or lf_node.name == 'argmin' or lf_node.name == 'sum':
        # expr -> Argmax|Sum(var variable, expr domain, expr body)

        prod = grammar.get_prod_by_ctr_name(lf_node.name.title())

        var_node = lf_node.children[0]
        var_field = RealizedField(prod['variable'], var_node.name)

        domain_node = lf_node.children[1]
        domain_ast_node = logical_form_to_ast(grammar, domain_node)
        domain_field = RealizedField(prod['domain'], domain_ast_node)

        body_node = lf_node.children[2]
        body_ast_node = logical_form_to_ast(grammar, body_node)
        body_field = RealizedField(prod['body'], body_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      [var_field, domain_field, body_field])
    elif lf_node.name == 'and' or lf_node.name == 'or':
        # expr -> And(expr* arguments) | Or(expr* arguments)
        prod = grammar.get_prod_by_ctr_name(lf_node.name.title())

        arg_ast_nodes = []
        for arg_node in lf_node.children:
            arg_ast_node = logical_form_to_ast(grammar, arg_node)
            arg_ast_nodes.append(arg_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      [RealizedField(prod['arguments'], arg_ast_nodes)])
    elif lf_node.name == 'not':
        # expr -> Not(expr argument)
        prod = grammar.get_prod_by_ctr_name('Not')

        arg_ast_node = logical_form_to_ast(grammar, lf_node.children[0])

        ast_node = AbstractSyntaxTree(prod,
                                      [RealizedField(prod['argument'], arg_ast_node)])
    elif lf_node.name == '>' or lf_node.name == '=' or lf_node.name == '<':
        # expr -> Compare(cmp_op op, expr left, expr right)
        prod = grammar.get_prod_by_ctr_name('Compare')
        op_name = 'GreaterThan' if lf_node.name == '>' else 'Equal' if lf_node.name == '=' else 'LessThan'
        op_field = RealizedField(prod['op'], AbstractSyntaxTree(grammar.get_prod_by_ctr_name(op_name)))

        left_node = lf_node.children[0]
        left_ast_node = logical_form_to_ast(grammar, left_node)
        left_field = RealizedField(prod['left'], left_ast_node)

        right_node = lf_node.children[1]
        right_ast_node = logical_form_to_ast(grammar, right_node)
        right_field = RealizedField(prod['right'], right_ast_node)

        ast_node = AbstractSyntaxTree(prod,
                                      [op_field, left_field, right_field])
    elif lf_node.name in ['jet', 'flight', 'from_airport', 'airport', 'airline', 'airline_name',
                          'class_type', 'aircraft_code', 'aircraft_code:t',
                          'from', 'to', 'day', 'month', 'year', 'arrival_time', 'limousine',
                          'departure_time', 'meal', 'meal:t', 'meal_code',
                          'during_day', 'tomorrow', 'daily', 'time_elapsed', 'time_zone_code',
                          'booking_class:t', 'booking_class', 'economy', 'ground_fare', 'class_of_service',
                          'capacity', 'weekday', 'today', 'turboprop', 'aircraft', 'air_taxi_operation',
                          'month_return', 'day_return', 'day_number_return', 'minimum_connection_time',
                          'during_day_arrival', 'connecting', 'minutes_distant',
                          'named', 'miles_distant', 'approx_arrival_time', 'approx_return_time',
                          'approx_departure_time', 'has_stops',
                          'day_after_tomorrow', 'manufacturer', 'discounted', 'overnight',
                          'nonstop', 'has_meal', 'round_trip', 'oneway', 'loc:t', 'ground_transport',
                          'to_city', 'flight_number', 'equals:t', 'abbrev', 'equals', 'rapid_transit',
                          'stop_arrival_time', 'arrival_month', 'cost',
                          'fare', 'services', 'fare_basis_code', 'rental_car', 'city', 'stop', 'day_number',
                          'days_from_today', 'after_day', 'before_day',
                          'airline:e', 'stops', 'month_arrival', 'day_number_arrival', 'day_arrival', 'taxi',
                          'next_days', 'restriction_code', 'tomorrow_arrival', 'tonight',
                          'population:i', 'state:t', 'next_to:t', 'elevation:i', 'size:i', 'capital:t',
                          'len:i', 'city:t', 'named:t', 'river:t', 'place:t', 'capital:c', 'major:t', 'town:t',
                          'mountain:t', 'lake:t', 'area:i', 'density:i', 'high_point:t', 'elevation:t', 'population:t',
                          'in:t']:
        # expr -> Apply(pred predicate, expr* arguments)
        prod = grammar.get_prod_by_ctr_name('Apply')

        pred_field = RealizedField(prod['predicate'], value=lf_node.name)

        arg_ast_nodes = []
        for arg_node in lf_node.children:
            arg_ast_node = logical_form_to_ast(grammar, arg_node)
            arg_ast_nodes.append(arg_ast_node)
        arg_field = RealizedField(prod['arguments'], arg_ast_nodes)

        ast_node = AbstractSyntaxTree(prod, [pred_field, arg_field])
    elif lf_node.name.startswith('$'):
        prod = grammar.get_prod_by_ctr_name('Variable')
        ast_node = AbstractSyntaxTree(prod,
                                      [RealizedField(prod['variable'], value=lf_node.name)])
    elif ':ap' in lf_node.name or ':fb' in lf_node.name or ':mf' in lf_node.name or \
                    ':me' in lf_node.name or ':cl' in lf_node.name or ':pd' in lf_node.name or \
                    ':dc' in lf_node.name or ':al' in lf_node.name or \
                    lf_node.name in ['yr0', 'do0', 'fb1', 'rc0', 'ci0', 'fn0', 'ap0', 'al1', 'al2', 'ap1', 'ci1',
                                     'ci2', 'ci3', 'st0', 'ti0', 'ti1', 'da0', 'da1', 'da2', 'da3', 'da4', 'al0',
                                     'fb0', 'dn0', 'dn1', 'mn0', 'ac0', 'fn1', 'st1', 'st2',
                                     'c0', 'm0', 's0', 'r0', 'n0', 'co0', 'usa:co', 'death_valley:lo', 's1',
                                     'colorado:n']:
        prod = grammar.get_prod_by_ctr_name('Entity')
        ast_node = AbstractSyntaxTree(prod,
                                      [RealizedField(prod['entity'], value=lf_node.name)])
    elif lf_node.name.endswith(':i') or lf_node.name.endswith(':hr'):
        prod = grammar.get_prod_by_ctr_name('Number')
        ast_node = AbstractSyntaxTree(prod,
                                      [RealizedField(prod['number'], value=lf_node.name)])
    elif lf_node.name == 'the':
        # expr -> The(var variable, expr body)
        prod = grammar.get_prod_by_ctr_name('The')

        var_node = lf_node.children[0]
        var_field = RealizedField(prod['variable'], var_node.name)

        body_node = lf_node.children[1]
        body_ast_node = logical_form_to_ast(grammar, body_node)
        body_field = RealizedField(prod['body'], body_ast_node)

        ast_node = AbstractSyntaxTree(prod, [var_field, body_field])
    elif lf_node.name == 'exists' or lf_node.name == 'max' or lf_node.name == 'min' or lf_node.name == 'count':
        # expr -> Exists(var variable, expr body)
        prod = grammar.get_prod_by_ctr_name(lf_node.name.title())

        var_node = lf_node.children[0]
        var_field = RealizedField(prod['variable'], var_node.name)

        body_node = lf_node.children[1]
        body_ast_node = logical_form_to_ast(grammar, body_node)
        body_field = RealizedField(prod['body'], body_ast_node)

        ast_node = AbstractSyntaxTree(prod, [var_field, body_field])
    else:
        raise NotImplementedError

    return ast_node


def ast_to_logical_form(ast_tree):
    constructor_name = ast_tree.production.constructor.name
    if constructor_name == 'Lambda':
        var_node = Node(ast_tree['variable'].value)
        type_node = Node(ast_tree['type'].value)
        body_node = ast_to_logical_form(ast_tree['body'].value)

        node = Node('lambda', [var_node, type_node, body_node])
    elif constructor_name in ['Argmax', 'Argmin', 'Sum']:
        var_node = Node(ast_tree['variable'].value)
        domain_node = ast_to_logical_form(ast_tree['domain'].value)
        body_node = ast_to_logical_form(ast_tree['body'].value)

        node = Node(constructor_name.lower(), [var_node, domain_node, body_node])
    elif constructor_name == 'Apply':
        predicate = ast_tree['predicate'].value
        arg_nodes = [ast_to_logical_form(tree) for tree in ast_tree['arguments'].value]

        node = Node(predicate, arg_nodes)
    elif constructor_name in ['Count', 'Exists', 'Max', 'Min', 'The']:
        var_node = Node(ast_tree['variable'].value)
        body_node = ast_to_logical_form(ast_tree['body'].value)

        node = Node(constructor_name.lower(), [var_node, body_node])
    elif constructor_name in ['And', 'Or']:
        arg_nodes = [ast_to_logical_form(tree) for tree in ast_tree['arguments'].value]

        node = Node(constructor_name.lower(), arg_nodes)
    elif constructor_name == 'Not':
        arg_node = ast_to_logical_form(ast_tree['argument'].value)

        node = Node('not', arg_node)
    elif constructor_name == 'Compare':
        op = {'GreaterThan': '>', 'Equal': '=', 'LessThan': '<'}[ast_tree['op'].value.production.constructor.name]
        left_node = ast_to_logical_form(ast_tree['left'].value)
        right_node = ast_to_logical_form(ast_tree['right'].value)

        node = Node(op, [left_node, right_node])
    elif constructor_name in ['Variable', 'Entity', 'Number']:
        node = Node(ast_tree.fields[0].value)
    else:
        raise ValueError('unknown AST node %s' % ast_tree)

    return node


if __name__ == '__main__':
    asdl_desc = """
    # define primitive fields
    var, ent, num, var_type, pred

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
    # lf = parse_lambda_expr('al0')
    for i, line in enumerate(chain(open('data/atis/train.txt'), open('data/atis/dev.txt'), open('data/atis/test.txt'))):
        line = line.strip()
        lf = parse_lambda_expr(line.split('\t')[1])
        ast_tree = logical_form_to_ast(grammar, lf)
        new_lf = ast_to_logical_form(ast_tree)
        assert lf == new_lf
        ast_tree.sanity_check()
        print(lf.to_string())
    pass
