# coding=utf-8

from __future__ import print_function

import token as tk
from cStringIO import StringIO
from tokenize import generate_tokens


def tokenize_code(code):
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break
        tokens.append(tokval)

    return tokens


if __name__ == '__main__':
    print(tokenize_code('offset = self.getpos()()'))
