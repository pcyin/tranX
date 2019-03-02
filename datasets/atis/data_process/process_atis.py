# -*- coding: utf-8 -*-
from __future__ import print_function

import random as rnd
import argparse
# import stat_atis
from .utils import *
from .utils_date_number import *


e2m_dict, m2e_dict, e2type_dict = read_entity_mention_mapping()


def q_process(_q):
  is_successful = True
  q = _q.strip().lower()
  # tokenize q
  q = add_padding(' '.join(norm_airline(norm_daynumber(
    norm_q_time(norm_dollar(q.split(' ')))))))

  # find entities in q, and replace them with type_id
  const_index_dict = {}
  type_index_dict = {}
  while True:
    q_list = list(filter(lambda x: len(x) > 0, ' '.join(map(lambda x: x, q.split(' '))).split(' ')))
    found_flag = False
    for n in range(7, 0, -1):
      if len(q_list) >= n:
        for i in range(0, len(q_list) - n + 1):
          m = ' '.join(q_list[i:i + n])
          should_replace_flag = True
          if m in m2e_dict:
            e = m2e_dict[m]
            t = e2type_dict[e]
            if (t == 'st' or is_state_token(m)) and (i > 0) and is_city_token(q_list[i - 1]):
              should_replace_flag = False
            elif (n == 1) and (m == 'may') and (i + 1 < len(q_list)) and (q_list[i + 1] == 'i'):
              should_replace_flag = False
          elif n == 1:
            if m.startswith('$'):
              e, t = m[1:] + ':do', 'do'
            elif m.startswith('_'):
              t, _e = m[1:].split('_')
              e = '%s:%s' % (_e, t)
            elif is_normalized_time_mention_str(m):
              e, t = convert_time_m2e(m), 'ti'
            else:
              should_replace_flag = False
          else:
            should_replace_flag = False
          if should_replace_flag:
            if e not in const_index_dict:
              type_index_dict[t] = type_index_dict.get(t, -1) + 1
              const_index_dict[e] = type_index_dict[t]
            if e not in e2type_dict:
              e2type_dict[e] = t
            q = q.replace(' %s ' % (m,), ' %s%d ' % (t, const_index_dict[e]))
            found_flag = True
            break
        if found_flag:
          break
    if not found_flag:
      break

  q = add_padding(fix_missing_link(q, const_index_dict, e2type_dict))

  q_list = list(filter(lambda x: len(x) > 0, ' '.join(
    map(lambda x: norm_word(x), q.split(' '))).split(' ')))

  return q_list, const_index_dict, type_index_dict

