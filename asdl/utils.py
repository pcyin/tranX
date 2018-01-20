# coding=utf-8
import re


def remove_comment(text):
    text = re.sub(re.compile("#.*"), "", text)
    text = '\n'.join(filter(lambda x: x, text.split('\n')))

    return text
