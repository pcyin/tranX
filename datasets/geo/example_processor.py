from datasets.utils import ExampleProcessor

from asdl.asdl_ast import AbstractSyntaxTree
from .data_process.process_geoquery import q_process


class GeoQueryExampleProcessor(ExampleProcessor):
    def __init__(self, transition_system):
        self.transition_system = transition_system

    def pre_process_utterance(self, utterance):
        canonical_utterance_tokens, const_index_dict, type_index_dict = q_process(utterance)

        slot2entity_map = dict()
        for typed_entity, idx in const_index_dict.items():
            entity_name, entity_type = typed_entity.split(':')
            slot2entity_map['%s%d' % (entity_type, idx)] = typed_entity

        return canonical_utterance_tokens, slot2entity_map

    def post_process_hypothesis(self, hyp, meta_info, utterance=None):
        """traverse the AST and replace slot ids with original strings"""
        slot2entity_map = meta_info

        def _travel(root):
                for field in root.fields:
                    if self.transition_system.grammar.is_primitive_type(field.type):
                        slot_name = field.value
                        if slot_name in slot2entity_map:
                            field.value = slot2entity_map[slot_name]
                    else:
                        for val in field.as_value_list:
                            _travel(val)

        _travel(hyp.tree)
        hyp.code = self.transition_system.ast_to_surface_code(hyp.tree)
