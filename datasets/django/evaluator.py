from components.evaluator import Evaluator
from common.registerable import Registrable
import ast
import astor


@Registrable.register('django_evaluator')
class DjangoEvaluator(Evaluator):
    def __init__(self, transition_system=None, args=None):
        super(DjangoEvaluator, self).__init__()
        self.transition_system = transition_system

    def is_hyp_correct(self, example, hyp):
        ref_code = example.tgt_code
        ref_py_ast = ast.parse(ref_code).body[0]
        ref_reformatted_code = astor.to_source(ref_py_ast).strip()

        ref_code_tokens = self.transition_system.tokenize_code(ref_reformatted_code)
        hyp_code_tokens = self.transition_system.tokenize_code(hyp.code)

        return ref_code_tokens == hyp_code_tokens
