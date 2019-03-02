# -*- coding: utf-8 -*-
from __future__ import print_function
import six
import sys
import json
import random as rnd
import argparse
import codecs
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()


def read_entity_mention_mapping(fn='data/geo/entity_mention.txt'):
  e2m_dict, m2e_dict, e2type_dict = {}, {}, {}

  with codecs.open(fn, encoding='utf-8') as f_in:
    for l in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f_in.read().split('\n'))):
      l_list = l.split('\t')
      e2m_dict[l_list[0]] = l_list[1:]
      e2type_dict[l_list[0]]=l_list[0].split(':')[-1]

  for k, v in e2m_dict.items():
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
  for n in range(2, 0, -1):
    if len(q_list) >= n:
      for i in range(0, len(q_list)-n+1):
        if (i+n>=len(q_list)) or (q_list[i+n]!='river'):
          m = ' '.join(q_list[i:i+n])+' river'
          if m in m2e_dict:
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
    q_list = list(filter(lambda x: len(x) > 0, ' '.join(map(lambda x: x, q.split(' '))).split(' ')))
    found_flag=False
    for n in range(5, 0, -1):
      if len(q_list) >= n:
        for i in range(0, len(q_list)-n+1):
          m = ' '.join(q_list[i:i+n])
          if m in m2e_dict:
            e = m2e_dict[m]
            t = e2type_dict[e]
            if e not in const_index_dict:
              type_index_dict[t] = type_index_dict.get(t, -1) + 1
              const_index_dict[e] = type_index_dict[t]
            q = q.replace(' %s ' % (m,), ' %s%d ' % (t, const_index_dict[e]))
            found_flag=True
            break
        if found_flag:
          break
    if not found_flag:
      break

  q_list = list(filter(lambda x: len(x) > 0, ' '.join(map(lambda x: norm_word(x), q.split(' '))).split(' ')))

  return q_list, const_index_dict, type_index_dict

