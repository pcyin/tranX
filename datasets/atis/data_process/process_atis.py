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
    q_list = filter(lambda x: len(x) > 0, ' '.join(map(lambda x: x, q.split(' '))).split(' '))
    found_flag = False
    for n in range(7, 0, -1):
      if len(q_list) >= n:
        for i in range(0, len(q_list) - n + 1):
          m = ' '.join(q_list[i:i + n])
          should_replace_flag = True
          if m2e_dict.has_key(m):
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
            if not const_index_dict.has_key(e):
              type_index_dict[t] = type_index_dict.get(t, -1) + 1
              const_index_dict[e] = type_index_dict[t]
            if not e2type_dict.has_key(e):
              e2type_dict[e] = t
            q = q.replace(' %s ' % (m,), ' %s%d ' % (t, const_index_dict[e]))
            found_flag = True
            break
        if found_flag:
          break
    if not found_flag:
      break

  q = add_padding(fix_missing_link(q, const_index_dict, e2type_dict))

  q_list = filter(lambda x: len(x) > 0, ' '.join(
    map(lambda x: norm_word(x), q.split(' '))).split(' '))

  return q_list, const_index_dict, type_index_dict


def qf_process(_q, _f, split):
  is_successful = True
  q, f = _q, _f
  # tokenize q
  q = add_padding(' '.join(norm_airline(norm_daynumber(
                 norm_q_time(norm_dollar(q.split(' ')))))))
  # tokenize f
  f = add_padding(' '.join([norm_form(it) for it in filter(lambda x: len(x) > 0, f.replace(
      '(', ' ( ').replace(')', ' ) ').strip().split(' '))]))

  if split == 'test':
    f = norm_lambda_variable(f)
  f = add_padding(fix_form_type_entity_mismatch(f))

  # find entities in q, and replace them with type_id
  const_index_dict = {}
  type_index_dict = {}
  while True:
    q_list = filter(lambda x: len(x) > 0, ' '.join(map(lambda x: x, q.split(' '))).split(' '))
    found_flag=False
    for n in range(7, 0, -1):
      if len(q_list) >= n:
        for i in range(0, len(q_list)-n+1):
          m = ' '.join(q_list[i:i+n])
          should_replace_flag=True
          if m2e_dict.has_key(m):
            e = m2e_dict[m]
            t = e2type_dict[e]
            if (t == 'st' or is_state_token(m)) and (i > 0) and is_city_token(q_list[i-1]):
              should_replace_flag = False
            elif (n == 1) and (m == 'may') and (i+1 < len(q_list)) and (q_list[i+1] == 'i'):
              should_replace_flag = False
          elif n == 1:
            if m.startswith('$'):
              e,t = m[1:]+':do','do'
            elif m.startswith('_'):
              t, _e = m[1:].split('_')
              e = '%s:%s' % (_e,t)
            elif is_normalized_time_mention_str(m):
              e,t = convert_time_m2e(m),'ti'
            else:
              should_replace_flag = False
          else:
            should_replace_flag = False
          if should_replace_flag:
            if not const_index_dict.has_key(e):
              type_index_dict[t] = type_index_dict.get(t, -1) + 1
              const_index_dict[e] = type_index_dict[t]
            if not e2type_dict.has_key(e):
              e2type_dict[e] = t
            q = q.replace(' %s ' % (m,), ' %s%d ' % (t, const_index_dict[e]))
            found_flag=True
            break
        if found_flag:
          break
    if not found_flag:
      break

  q = add_padding(fix_missing_link(q, const_index_dict, e2type_dict))

  # replace const entity with ``type_id''
  for e_name, e_type in sort_entity_list(entity_re.findall(f.replace(' ', '  '))):
    const_str='%s:%s' % (e_name, e_type)
    if const_index_dict.has_key(const_str):
      f = f.replace(' %s ' % (const_str,), ' %s%d ' % (e2type_dict[const_str], const_index_dict[const_str]))
    else:
      if e_type in set(('ci','ap','ti','rc','mn','dn')):
        is_successful = False

  q_list = filter(lambda x: len(x) > 0, ' '.join(
      map(lambda x: norm_word(x), q.split(' '))).split(' '))
  f_list = filter(lambda x: len(x) > 0, f.strip().split(' '))

  return (q_list, f_list, const_index_dict, type_index_dict, is_successful)


def process_main(d, split):
  with open('data/atis/%s.raw' % (split,), 'r') as f_in:
    raw_list = filter(lambda x: len(x) > 0, map(
        lambda x: x.strip(), f_in.read().decode('utf-8').split('\n')))
    qf_list = [(raw_list[2 * i], raw_list[2 * i + 1])
               for i in xrange(len(raw_list) / 2)]
  l_list = []
  for q, f in qf_list:
    q_list, f_list, is_successful = qf_process(q, f, split)
    if ((split != 'test') and is_successful) or (split == 'test'):
      q_orig, f_orig, _ = stat_atis.qf_process(q, f, split)
      q_orig_length, f_orig_length = len(q_orig), len(f_orig)
      l_list.append((q_list, f_list, q_orig_length, f_orig_length))
  l_list.sort(key=lambda x: len(x[0]))
  with open(d + '%s.txt' % (split,), 'w') as f_out:
    f_out.write('\n'.join(map(lambda x: '%s\t%s' %
                              (' '.join(x[0]), ' '.join(x[1])), l_list)).encode('utf-8'))
    print('maximun length:', max([len(x[0]) for x in l_list]), max([len(x[1]) for x in l_list]))

def vocab_main(d):
  cq, cf = {}, {}
  l_list = []
  with open(d + 'train.txt', 'r') as f_in:
    l_list.extend(filter(lambda x: len(x) > 0, map(
        lambda x: x.strip(), f_in.read().decode('utf-8').split('\n'))))
  for l in l_list:
    q, f = l.decode('utf-8').strip().split('\t')
    for it in q.split(' '):
      cq[it] = cq.get(it, 0) + 1
    for it in f.split(' '):
      cf[it] = cf.get(it, 0) + 1
  with open(d + 'vocab.q.txt', 'w') as f_out:
    for w, c in sorted([(k, v) for k, v in cq.iteritems()], key=lambda x: x[1], reverse=True):
      print(('%s\t%d' % (w, c)).encode('utf-8'), file=f_out)
  with open(d + 'vocab.f.txt', 'w') as f_out:
    for w, c in sorted([(k, v) for k, v in cf.iteritems()], key=lambda x: x[1], reverse=True):
      print(('%s\t%d' % (w, c)).encode('utf-8'), file=f_out)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--split", help="official split",
                      default=1)
  parser.add_argument("-d", "--data_dir", help="input folder",
                      default="/disk/scratch_ssd/lidong/deep_qa/atis/")
  args = parser.parse_args()

  for split in ('train', 'dev', 'test'):
    print(split, ':')
    process_main(args.data_dir, split)

  vocab_main(args.data_dir)
