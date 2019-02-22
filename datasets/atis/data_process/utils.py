# -*- coding: utf-8 -*-

import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from .misspellings import _misspelling_dict
from rex import rex

stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()
entity_re = re.compile(r' (\w+?):(\w+?) ')


def read_airline_mapping(fn='data/atis/airline_code.txt'):
  airline_word = set(('airline', 'airlines', 'air',
                      'airways', 'flights', 'flight'))
  # airline code crawled from
  # http://vankata.be/aviationbg/En/Info/Airline_cod.htm
  al_dict = {}
  al_set = set(('aa', 'ua', 'us', 'co', 'nw', 'dl', 'tw', 'cp', 'ac', 'yx',
                'ea', 'hp', 'as', 'nx', 'ir', 'ta', 'wn', 'ff', 'ml', 'al', 'lh', 'kw', 'ji'))
  with open(fn, 'r') as f_in:
    for l in filter(lambda x: len(x) > 0, map(lambda x: x.strip().lower(), f_in.read().decode('utf-8').split('\n'))):
      l_list = l.split('\t')
      if l_list[0] in al_set:
        for it in l_list[1:]:
          al_dict[it] = l_list[0]
          al_dict[it + "'s"] = l_list[0]
          it_list = it.split(' ')
          if it_list[-1] in airline_word:
            for w in airline_word:
              al_dict[' '.join(it_list[:-1]) + ' ' + w] = l_list[0]
        for w in airline_word:
          al_dict[l_list[0] + ' ' + w] = l_list[0]
        al_dict[l_list[0]] = l_list[0]
  return al_dict

al_dict = read_airline_mapping()


def sort_entity_list(e_list):
  return sorted([e.split('\t') for e in set(['\t'.join((n, t)) for n, t in e_list])], key=lambda x: x[1])


def read_iata_mapping(e2m_dict, e2type_dict, fn='data/atis/iata.txt'):
  # IATA code crawled from wikipedia
  with open(fn, 'r') as f_in:
    for l in filter(lambda x: len(x) > 0, map(lambda x: x.strip().lower(), f_in.read().decode('utf-8').split('\n'))):
      l_list = l.split('\t')
      k = '%s:ap' % (l_list[0],)
      e2m_dict[k] = l_list[1:]
      if l_list[1].endswith('international airport'):
        e2m_dict[k].append(
            ' '.join(l_list[1].split(' ')[:-2]) + ' international')
        e2m_dict[k].append(' '.join(l_list[1].split(' ')[:-2]) + ' airport')
        e2m_dict[k].append(' '.join(l_list[1].split(' ')[:-2]) + '\'s airport')
      e2m_dict[k].append(l_list[0])
      e2m_dict[k] = sorted(list(set(e2m_dict[k])),
                           key=lambda x: len(x), reverse=True)
      e2type_dict[k] = 'ap'


def read_entity_mention_mapping(fn='data/atis/entity_mention.txt'):
  e2m_dict, m2e_dict, e2type_dict = {}, {}, {}

  with open(fn, 'r') as f_in:
    for l in filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f_in.read().decode('utf-8').split('\n'))):
      l_list = l.split('\t')
      e2m_dict[l_list[0]] = l_list[1:]
      e2type_dict[l_list[0]] = l_list[0].split(':')[-1]

  read_iata_mapping(e2m_dict, e2type_dict)

  for k, v in e2m_dict.iteritems():
    for it in v:
      m2e_dict[it]=k

  return e2m_dict, m2e_dict, e2type_dict


def norm_word(w):
  if w == None or len(w) == 0:
    return ''
  w = w.lower()
  if _misspelling_dict.has_key(w):
    w = _misspelling_dict[w][0]

  w_stem = stemmer.stem(w)

  return w_stem


def norm_form(w):
  return w


def cannot_follow_as(w):
  return (w in set(('soon','well'))) or w.startswith('earl') or w.startswith('possib')


def norm_airline(_w_list):
  w_list = []
  i = 0
  while i < len(_w_list):
    w = _w_list[i]
    if (i + 2 < len(_w_list)) and al_dict.has_key(' '.join(_w_list[i:i + 3])):
      w_list.append('_al_' + al_dict[' '.join(_w_list[i:i + 3])])
      i += 2
    elif (i + 1 < len(_w_list)) and al_dict.has_key(' '.join(_w_list[i:i + 2])):
      w_list.append('_al_' + al_dict[' '.join(_w_list[i:i + 2])])
      i += 1
    elif al_dict.has_key(w):
      if (w == 'as') and (i + 1 < len(_w_list)) and cannot_follow_as(_w_list[i+1]):
        w_list.append(w)
      else:
        w_list.append('_al_' + al_dict[w])
    else:
      w_list.append(w)
    i += 1

  return w_list


def norm_lambda_variable(f):
  f_list = f.strip().split(' ')
  v_dict = {}
  for i in range(len(f_list)):
    w = f_list[i]
    if w.startswith('$') or ((i - 1 >= 0) and (f_list[i - 1] == 'lambda')):
      if not v_dict.has_key(w):
        v_dict[w] = '$%d' % (len(v_dict),)

  # missing ')'
  c_left = f_list.count('(')
  c_right = f_list.count(')')
  if c_right < c_left:
    f_list.extend([')' for it in range(c_left - c_right)])
  elif c_right > c_left:
    print(c_right, '>', c_left, f_list)

  return ' '.join(map(lambda x: v_dict.get(x, x), f_list))


def read_ci2ap_dict(fn='data/atis/ci_ap_mapping.txt'):
  ci2ap_dict = {}
  with open(fn,'r') as f_in:
    for l in f_in:
      ci,ap=l.decode('utf-8').strip().split('\t')
      ci2ap_dict[ci]=ap
  ap2ci_dict = dict([(v,k) for k,v in ci2ap_dict.items()])
  return ci2ap_dict, ap2ci_dict
ci2ap_dict, ap2ci_dict=read_ci2ap_dict()


def fix_form_type_entity_mismatch(f):
  f_list = filter(lambda x: len(x) > 0, f.strip().split(' '))
  for i in range(0, len(f_list) - 2):
    if (f_list[i]=='from_airport') and f_list[i+1].startswith('$') and f_list[i+2].endswith(':ci'):
      f_list[i+2]=ci2ap_dict.get(f_list[i+2], f_list[i+2])
  return ' '.join(f_list)


def is_city_token(t):
  return t.startswith('ci') and (len(t)>2) and t[2:].isdigit()


def is_state_token(t):
  return t in set(('washington',))


def rex_list(re_list, q):
  m = None
  for re_it in re_list:
    m = rex(re_it, q)
    if m:
      break
  return m

def fix_missing_link(q, const_index_dict, e2type_dict):
  transport_re_list = ("/((between)|(from)) (the ){0,1}(?P<m1>.+?) ((and)|((in){0,1}to)) (the ){0,1}(?P<m2>.+) in/", "/((between)|(from)) (the ){0,1}(?P<m1>.+?) ((and)|((in){0,1}to)) (the ){0,1}(?P<m2>.+)/",)
  ci_flag, ap_flag = ('ci0' in q), ('ap0' in q)
  found_flag = False
  if ((' transport' in q) or (' distance' in q) or (' how far ' in q) or rex("/((go)|(get)) from (the ){0,1}((downtown)|(town)|(airport)|(ap0)|(ci0))( in ci0){0,1} (in){0,1}to (the ){0,1}((downtown)|(town)|(airport)|(ap0)|(ci0))/", q)) and (ci_flag != ap_flag):
    m = rex_list(transport_re_list, q)
    if m:
      for it1 in ('m1', 'm2'):
        if m[it1] == 'airport in ci0':
          q = q.replace(' airport in ci0 ', ' ap0 ')
          m = rex_list(transport_re_list, q)
      for it1,it12 in (('ci0', 'ap0'), ('ap0', 'ci0')):
        for it2, it22 in (('m1', 'm2'), ('m2', 'm1')):
          if (it1 in m[it2]):
            found_flag = True
            if not(it12 in m[it22]):
              q = q.replace(' ' + m[it22], ' %s %s' % (it12, m[it22]))
            break
        if found_flag:
          break
      if not found_flag:
        for it1 in ('m1', 'm2'):
          for it2, it22 in ((' airport', 'ap0'), (' downtown', 'ci0'), (' town', 'ci0')):
            if (it2 in (' '+m[it1])):
              found_flag = True
              q = q.replace(' ' + m[it1], ' %s %s' % (it22, m[it1]))
              break
  if found_flag:
    for const_str, v in const_index_dict.items():
      if (v == 0) and (e2type_dict[const_str] == ('ci' if ci_flag else 'ap')):
        const_str_fix = (ci2ap_dict[const_str] if ci_flag else ap2ci_dict[const_str])
        const_index_dict[const_str_fix], e2type_dict[const_str_fix] = 0, ('ap' if ci_flag else 'ci')
        break
  return q


def add_padding(q):
  return ' ' + q + ' '
