# coding=utf-8

from __future__ import print_function

import token as tk
try:
    from cStringIO import StringIO
except:
    from io import StringIO
from tokenize import generate_tokens


def tokenize_code(code, mode=None):
    token_stream = generate_tokens(StringIO(code).readline)
    tokens = []
    for toknum, tokval, (srow, scol), (erow, ecol), _ in token_stream:
        if toknum == tk.ENDMARKER:
            break

        if mode == 'decoder':
            if toknum == tk.STRING:
                quote = tokval[0]
                tokval = tokval[1:-1]
                tokens.append(quote)
                tokens.append(tokval)
                tokens.append(quote)
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        elif mode == 'canonicalize':
            if toknum == tk.STRING:
                tokens.append('_STR_')
            elif toknum == tk.DEDENT:
                continue
            else:
                tokens.append(tokval)
        else:
            tokens.append(tokval)

    return tokens


if __name__ == '__main__':
    print(tokenize_code('offset = self.getpos()()'))
