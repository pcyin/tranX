import astor

from datasets.utils import ExampleProcessor
from datasets.conala.dataset import canonicalize_intent, tokenize_intent, asdl_ast_to_python_ast, decanonicalize_code


class ConalaExampleProcessor(ExampleProcessor):
    def __init__(self, transition_system):
        self.transition_system = transition_system

    def pre_process_utterance(self, utterance):
        canonical_intent, slot_map = canonicalize_intent(utterance)
        intent_tokens = tokenize_intent(canonical_intent)

        return intent_tokens, slot_map

    def post_process_hypothesis(self, hyp, meta_info, utterance=None):
        """traverse the AST and replace slot ids with original strings"""
        hyp_ast = asdl_ast_to_python_ast(hyp.tree, self.transition_system.grammar)
        code_from_hyp = astor.to_source(hyp_ast).strip()
        hyp.code = decanonicalize_code(code_from_hyp, meta_info)
