# coding=utf-8


class ExampleProcessor(object):
    """
    Process a raw input utterance using domain-specific procedures (e.g., stemming),
    and post-process a generated hypothesis to the final form
    """
    def pre_process_utterance(self, utterance):
        raise NotImplementedError

    def post_process_hypothesis(self, hyp, meta_info, **kwargs):
        raise NotImplementedError


def get_example_processor_cls(dataset):
    if dataset == 'django':
        from datasets.django.example_processor import DjangoExampleProcessor
        return DjangoExampleProcessor
    elif dataset == 'atis':
        from datasets.atis.example_processor import ATISExampleProcessor
        return ATISExampleProcessor
    elif dataset == 'geo':
        from datasets.geo.example_processor import GeoQueryExampleProcessor
        return GeoQueryExampleProcessor
    elif dataset == 'conala':
        from datasets.conala.example_processor import ConalaExampleProcessor
        return ConalaExampleProcessor
    else:
        raise RuntimeError()
