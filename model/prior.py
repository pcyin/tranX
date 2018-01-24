# coding=utf-8


class Prior(object):
    def __init__(self):
        pass

    def __call__(self, code_list):
        raise NotImplementedError


class UniformPrior(Prior):
    def __call__(self, code_list):
        return [0. for code in code_list]
