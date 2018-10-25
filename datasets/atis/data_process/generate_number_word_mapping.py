# -*- coding: utf-8 -*-

from num2words import num2words


def generate_variant(w):
  r_list = [w]
  if w.find('-') >= 0:
    r_list.append(w.replace('-', ''))
  return r_list

with open('../data/atis/number_word_mapping.txt', 'w') as f_out:
  for i in xrange(1, 31 + 1):
    print >>f_out, '\t'.join(['\t'.join(it) for it in ([str(i)], generate_variant(
        num2words(i)),	generate_variant(num2words(i, ordinal=True)))]).encode('utf-8')
