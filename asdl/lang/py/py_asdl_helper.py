# coding=utf-8

import sys

from asdl.asdl_ast import RealizedField, AbstractSyntaxTree


# from https://stackoverflow.com/questions/15357422/python-determine-if-a-string-should-be-converted-into-int-or-float
def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def python_ast_to_asdl_ast(py_ast_node, grammar):
    # node should be composite
    py_node_name = type(py_ast_node).__name__
    # assert py_node_name.startswith('_ast.')

    production = grammar.get_prod_by_ctr_name(py_node_name)

    fields = []
    for field in production.fields:
        field_value = getattr(py_ast_node, field.name)
        asdl_field = RealizedField(field)
        if field.cardinality == 'single' or field.cardinality == 'optional':
            if field_value is not None:  # sometimes it could be 0
                if grammar.is_composite_type(field.type):
                    child_node = python_ast_to_asdl_ast(field_value, grammar)
                    asdl_field.add_value(child_node)
                else:
                    asdl_field.add_value(str(field_value))
        # field with multiple cardinality
        elif field_value is not None:
                if grammar.is_composite_type(field.type):
                    for val in field_value:
                        child_node = python_ast_to_asdl_ast(val, grammar)
                        asdl_field.add_value(child_node)
                else:
                    for val in field_value:
                        asdl_field.add_value(str(val))

        fields.append(asdl_field)

    asdl_node = AbstractSyntaxTree(production, realized_fields=fields)

    return asdl_node


def asdl_ast_to_python_ast(asdl_ast_node, grammar):
    py_node_type = getattr(sys.modules['ast'], asdl_ast_node.production.constructor.name)
    py_ast_node = py_node_type()

    for field in asdl_ast_node.fields:
        # for composite node
        field_value = None
        if grammar.is_composite_type(field.type):
            if field.value and field.cardinality == 'multiple':
                field_value = []
                for val in field.value:
                    node = asdl_ast_to_python_ast(val, grammar)
                    field_value.append(node)
            elif field.value and field.cardinality in ('single', 'optional'):
                field_value = asdl_ast_to_python_ast(field.value, grammar)
        else:
            # for primitive node, note that primitive field may have `None` value
            if field.value is not None:
                if field.type.name == 'object':
                    if '.' in field.value or 'e' in field.value:
                        field_value = float(field.value)
                    elif isint(field.value):
                        field_value = int(field.value)
                    else:
                        raise ValueError('cannot convert [%s] to float or int' % field.value)
                elif field.type.name == 'int':
                    field_value = int(field.value)
                else:
                    field_value = field.value

            # FIXME: hack! if int? is missing value in ImportFrom(identifier? module, alias* names, int? level), fill with 0
            elif field.name == 'level':
                field_value = 0

        # must set unused fields to default value...
        if field_value is None and field.cardinality == 'multiple':
            field_value = list()

        setattr(py_ast_node, field.name, field_value)

    return py_ast_node