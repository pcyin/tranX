# -*- coding: utf-8 -*-

import json
import random as rnd
import argparse
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()


def read_entity_mention_mapping(fn='data/geo/entity_mention.txt'):
  e2m_dict, m2e_dict, e2type_dict = {}, {}, {}

  with open(fn, 'r') as f_in:
    for l in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f_in.read().decode('utf-8').split('\n'))):
      l_list = l.split('\t')
      e2m_dict[l_list[0]] = l_list[1:]
      e2type_dict[l_list[0]]=l_list[0].split(':')[-1]

  for k, v in e2m_dict.iteritems():
    for it in v:
      m2e_dict[it]=k

  return e2m_dict, m2e_dict, e2type_dict

e2m_dict, m2e_dict, e2type_dict = read_entity_mention_mapping()


def norm_word(w):
  if w == None or len(w) == 0:
    return ''
  w_lem = lemmer.lemmatize(w)
  w_stem = stemmer.stem(w)
  if w_lem in set(('american', 'continental', 'usa')):
    return ''
  if w_lem in set(('pas',)):
    return stemmer.stem('pass')
  if w_lem in set(('though',)):
    return stemmer.stem('through')
  if w_lem in set(('much',)):
    return stemmer.stem('many')
  if w_lem in set(('count',)):
    return stemmer.stem('how many')
  if w_lem in set(('contains',)):
    return stemmer.stem('contain')
  if w_lem in set(('cite',)):
    return stemmer.stem('city')
  if w_lem in set(('whats',)):
    return stemmer.stem('what is')
  if w_lem in set(('peak',)):
    return stemmer.stem('mountain')
  if w_lem in set(('neighbor', 'adjoin', 'surround')):
    return stemmer.stem('border')
  if w_lem in set(('tallest', 'maximum')):
    return stemmer.stem('highest')
  if w_lem in set(('resident', 'inhabitant')):
    return stemmer.stem('people')
  if w_lem in set(('reside', 'stay', 'lived')):
    return stemmer.stem('live')
  if w_lem in set(('called',)):
    return stemmer.stem('named')
  if w_lem in set(('spot',)):
    return stemmer.stem('point')
  if w_lem in set(('large',)):
    return stemmer.stem('big')

  if w_stem in set(('adjac',)):
    return 'neighbor'
  if w_stem in set(('sparsest',)):
    return 'smallest'
  if w_stem in set(('tall',)):
    return 'high'
  if w_stem in set(('longer', 'lower', 'higher')):
    return '-er'

  return w_stem


def norm_form(w):
  return w


def sort_entity_list(e_list):
  return sorted([e.split('\t') for e in set(['\t'.join((n, t)) for n, t in e_list])], key=lambda x: x[1])

entity_re = re.compile(r' (\w+?):(\w+?) ')


def river_name(q_list):
  for n in xrange(2, 0, -1):
    if len(q_list) >= n:
      for i in xrange(0, len(q_list)-n+1):
        if (i+n>=len(q_list)) or (q_list[i+n]!='river'):
          m = ' '.join(q_list[i:i+n])+' river'
          if m2e_dict.has_key(m):
            # maybe a river name
            river_flag=False
            if ((i+n<len(q_list)) and (q_list[i+n].startswith('run') or q_list[i+n].startswith('traver') or q_list[i+n].startswith('flow'))) or (' '.join(q_list).startswith('how long ')) or ((i-1>=0) and q_list[i-1]=='the'):
              river_flag=True
            if river_flag:
              q_list.insert(i+n,'river')
              return q_list
  return q_list


def q_process(_q):
  is_successful = True
  q = _q
  # tokenize q
  q = ' ' + ' '.join(river_name(q.split(' '))) + ' '

  # find entities in q, and replace them with type_id
  const_index_dict = {}
  type_index_dict = {}
  while True:
    q_list = filter(lambda x: len(x) > 0, ' '.join(map(lambda x: x, q.split(' '))).split(' '))
    found_flag=False
    for n in xrange(5, 0, -1):
      if len(q_list) >= n:
        for i in xrange(0, len(q_list)-n+1):
          m = ' '.join(q_list[i:i+n])
          if m2e_dict.has_key(m):
            e = m2e_dict[m]
            t = e2type_dict[e]
            if not const_index_dict.has_key(e):
              type_index_dict[t] = type_index_dict.get(t, -1) + 1
              const_index_dict[e] = type_index_dict[t]
            q = q.replace(' %s ' % (m,), ' %s%d ' % (t, const_index_dict[e]))
            found_flag=True
            break
        if found_flag:
          break
    if not found_flag:
      break

  q_list = filter(lambda x: len(x) > 0, ' '.join(
      map(lambda x: norm_word(x), q.split(' '))).split(' '))

  return q_list, const_index_dict, type_index_dict


def qf_process(_q, _f):
  is_successful = True
  q, f = _q, _f
  # tokenize q
  q = ' ' + ' '.join(river_name(q.split(' '))) + ' '
  # tokenize f
  f = ' ' + ' '.join([norm_form(it) for it in filter(lambda x: len(x) > 0, f.replace(
      '(', ' ( ').replace(')', ' ) ').strip().split(' '))]) + ' '

  # find entities in q, and replace them with type_id
  const_index_dict = {}
  type_index_dict = {}
  while True:
    q_list = filter(lambda x: len(x) > 0, ' '.join(map(lambda x: x, q.split(' '))).split(' '))
    found_flag=False
    for n in xrange(5, 0, -1):
      if len(q_list) >= n:
        for i in xrange(0, len(q_list)-n+1):
          m = ' '.join(q_list[i:i+n])
          if m2e_dict.has_key(m):
            e = m2e_dict[m]
            t = e2type_dict[e]
            if not const_index_dict.has_key(e):
              type_index_dict[t] = type_index_dict.get(t, -1) + 1
              const_index_dict[e] = type_index_dict[t]
            q = q.replace(' %s ' % (m,), ' %s%d ' % (t, const_index_dict[e]))
            found_flag=True
            break
        if found_flag:
          break
    if not found_flag:
      break

  # replace const entity with ``type_id''
  for e_name, e_type in sort_entity_list(entity_re.findall(f.replace(' ', '  '))):
    const_str='%s:%s' % (e_name, e_type)
    const_id=const_str
    if const_index_dict.has_key(const_str):
      link_index = const_index_dict[const_str]
      f = f.replace(' %s ' % (const_id,), ' %s%d ' % (e2type_dict[const_str], link_index))
    # else:
    #   # this entity is not found in q
    #   link_index = -1
    #   print f, const_str
    #   print q, const_id, ' %s%d ' % (const_str, link_index)

  q_list = filter(lambda x: len(x) > 0, ' '.join(
      map(lambda x: norm_word(x), q.split(' '))).split(' '))
  f_list = filter(lambda x: len(x) > 0, f.strip().split(' '))

  return (q_list, f_list, is_successful)


def process_main(d, split):
  with open('data/geoqueries/%s.raw' % (split,), 'r') as f_in:
    raw_list = filter(lambda x: (len(x) > 0) and (not x.startswith('//')), map(
        lambda x: x.strip(), f_in.read().decode('utf-8').split('\n')))
    qf_list = [(raw_list[2 * i], raw_list[2 * i + 1])
               for i in xrange(len(raw_list) / 2)]
  l_list = []
  for q, f in qf_list:
    q_list, f_list, is_successful = qf_process(q, f)
    if ((split != 'test') and is_successful) or (split == 'test'):
      l_list.append((q_list, f_list))
  l_list.sort(key=lambda x: len(x[0]))
  with open(d + '%s.txt' % (split,), 'w') as f_out:
    f_out.write('\n'.join(map(lambda x: '%s\t%s' %
                              (' '.join(x[0]), ' '.join(x[1])), l_list)).encode('utf-8'))
    print 'maximun length:', max([len(x[0]) for x in l_list]), max([len(x[1]) for x in l_list])


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
      print >>f_out, ('%s\t%d' % (w, c)).encode('utf-8')
  with open(d + 'vocab.f.txt', 'w') as f_out:
    for w, c in sorted([(k, v) for k, v in cf.iteritems()], key=lambda x: x[1], reverse=True):
      print >>f_out, ('%s\t%d' % (w, c)).encode('utf-8')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data_dir", help="input folder",
                      default="/disk/scratch_ssd/lidong/deep_qa/geoqueries/")
  args = parser.parse_args()

  for split in ('train', 'test'):
    print split, ':'
    process_main(args.data_dir, split)

  vocab_main(args.data_dir)
