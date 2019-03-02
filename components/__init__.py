import six

from datasets.django.evaluator import DjangoEvaluator

if six.PY3:
    from datasets.conala.evaluator import ConalaEvaluator
    from datasets.wikisql.evaluator import WikiSQLEvaluator

