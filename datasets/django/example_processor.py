import astor

from common.registerable import Registrable
from datasets.utils import ExampleProcessor
from datasets.django.dataset import Django, replace_string_ast_nodes
from asdl.lang.py.py_asdl_helper import asdl_ast_to_python_ast


@Registrable.register('django_example_processor')
class DjangoExampleProcessor(ExampleProcessor):
    def __init__(self, transition_system):
        self.transition_system = transition_system

    def pre_process_utterance(self, utterance):
        canonical_utterance, str2slot_map = Django.canonicalize_query(utterance)

        meta_info = str2slot_map
        return canonical_utterance.split(' '), meta_info

    def post_process_hypothesis(self, hyp, meta_info, utterance=None):
        """traverse the AST and replace slot ids with original strings"""
        slot2str_map = {v: k for k, v in meta_info.items()}
        hyp_ast = asdl_ast_to_python_ast(hyp.tree, self.transition_system.grammar)
        replace_string_ast_nodes(hyp_ast, slot2str_map)

        hyp.code = astor.to_source(hyp_ast).strip()