# -*- coding: utf-8 -*-

import re


word_am_pm_re = re.compile(r'\d{1,4}[ap]m$')
digit_re = re.compile(r'\d{1,4}$')
time_between_and_re = re.compile(r'between [0-9]+\w* and')
time_from_to_re = re.compile(r'from [0-9]+\w* to')


def is_time_digit(w):
  if digit_re.match(w):
    if len(w) == 1 or len(w) == 2:
      return int(w) < 24
    elif len(w) == 3:
      return int(w[0]) < 24
    elif len(w) == 4:
      return int(w[:2]) < 24
  return False


def no_am_indicator(w_list, i):
  for w in w_list:
    if w in ('morning', 'am'):
      return False
    elif w in ('afternoon', 'pm', 'evening'):
      return True
  return True


def norm_q_time(_w_list):
  is_pm = True

  w_list = []
  for i in xrange(len(_w_list)):
    w = _w_list[i]
    if w in ('noon', 'noontime', 'dinnertime'):
      w_list.append('1200pm')
    elif (w == 'lunch') and (i >= 1) and _w_list[i - 1] == 'after':
      w_list.append('1400pm')
    elif w in ('midnight',):
      w_list.append('0000am')
    else:
      w_list.append(w)

  _w_list = w_list
  w_list = []
  for i in xrange(len(_w_list)):
    w = _w_list[i]
    if (w in ('12', '12pm', '1200pm')) and (i + 1 < len(_w_list)) and _w_list[i + 1] == '1200pm':
      pass
    elif (((i >= 1) and (_w_list[i - 1] in ('around', 'before', 'after', 'at', 'morning', 'afternoon'))) or ((i + 1 < len(_w_list)) and (_w_list[i + 1] in ("o'clock",))) or ((i + 3 < len(_w_list)) and (' '.join(_w_list[i + 1:i + 4]) in ('in the afternoon', 'in the morning')))) and is_time_digit(w):
      if len(w) == 4:
        is_pm = (int(w[:2]) >= 12)
      elif (len(w) == 2) and (int(w[:2]) >= 12):
        is_pm = True
      else:
        is_pm = no_am_indicator(_w_list, i)
      w_list.append(w + ('pm' if is_pm else 'am'))
    elif ((i + 3 < len(_w_list)) and w == 'one' and (' '.join(_w_list[i + 1:i + 4]) == 'in the afternoon')):
      w_list.append('1300pm')
    elif ((i - 1 >= 0) and (_w_list[i - 1] == 'between') and (i + 2 < len(_w_list)) and (_w_list[i + 1] == 'and') and time_between_and_re.match(' '.join(_w_list[i - 1:i + 1 + 1]))) or ((i - 1 >= 0) and (_w_list[i - 1] == 'from') and (i + 2 < len(_w_list)) and (_w_list[i + 1] == 'to') and time_from_to_re.match(' '.join(_w_list[i - 1:i + 1 + 1]))):
      if w.endswith('am') or w.endswith('pm'):
        w_list.append(w)
      elif is_time_digit(w):
        if len(w) == 4:
          is_pm = (int(w[:2]) >= 12)
        elif (len(w) == 2) and (int(w[:2]) >= 12):
          is_pm = True
        elif _w_list[i + 2].endswith('am'):
          is_pm = False
        else:
          is_pm = _w_list[i + 2].endswith('pm') or no_am_indicator(_w_list, i)
        w_list.append(w + ('pm' if is_pm else 'am'))
      else:
        print '>> between_and:', ' '.join(_w_list)
        w_list.append(w)

      if not (_w_list[i + 2].endswith('am') or _w_list[i + 2].endswith('pm')):
        if is_time_digit(_w_list[i + 2]) and (w_list[len(w_list) - 1][-2:] in ('am', 'pm')):
          _w_list[i + 2] = _w_list[i + 2] + w_list[len(w_list) - 1][-2:]
        else:
          print '>> between_and:', ' '.join(_w_list)
    else:
      w_list.append(w)

  _w_list = w_list
  w_list = []
  for w in _w_list:
    if word_am_pm_re.match(w):
      digit_str = w[:-2]
      # only hour
      if len(digit_str) <= 2:
        digit_str += '00'
      # \d\d\d\d[ap]m
      digit_str = ''.join(
          ['0' for i in xrange(4 - len(digit_str))]) + digit_str
      if w[-2:] == 'pm':
        h = int(digit_str[:2])
        if h < 12:
          digit_str = str(h + 12) + digit_str[2:]
      if w[-2:] == 'am' and w[:2] == '12':
        digit_str = '00' + digit_str[-2:]
      w_list.append(digit_str + w[-2:])
    else:
      w_list.append(w)

  return w_list

dollar_digit_re = re.compile(r'\d+$')


def norm_dollar(_w_list):
  w_list = []
  i = 0
  while i < len(_w_list):
    w = _w_list[i]
    if (i + 1 < len(_w_list)) and _w_list[i + 1].startswith('dollar') and dollar_digit_re.match(w):
      w_list.append('$' + w)
      i += 1
    elif (((i - 1 >= 0) and (_w_list[i - 1] == 'under')) or ((i - 2 >= 0) and (_w_list[i - 2] == 'less') and (_w_list[i - 1] == 'than'))) and dollar_digit_re.match(w):
      w_list.append('$' + w)
    else:
      w_list.append(w)
    i += 1

  return w_list


def norm_time_mention_str(e_name):
  if len(e_name) < 4:
    e_name = ''.join(
        ['0' for i in xrange(4 - len(e_name))]) + e_name
  e_name += ('pm' if int(e_name[:2]) >= 12 else 'am')
  return e_name


def convert_time_m2e(m):
  if m[:4]=='0000':
    return '0:ti'
  else:
    return m[:4].lstrip('0') + ':ti'


month_set = set(('january', 'march', 'july', 'september', 'may', 'december',
                 'november', 'june', 'february', 'october', 'april', 'august'))
week_set = set(('wednesday', 'sunday', 'friday', 'monday',
                'thursday', 'tuesday', 'saturday'))


def read_daynumber_word_mapping(fn='data/atis/number_word_mapping.txt'):
  word2num = {}
  with open(fn, 'r') as f_in:
    for l in filter(lambda x: len(x) > 0, map(lambda x: x.strip().lower(), f_in.read().decode('utf-8').split('\n'))):
      l_list = l.split('\t')
      for w in l_list[1:]:
        word2num[w] = l_list[0]
  return word2num

word2num = read_daynumber_word_mapping()


def norm_daynumber(_w_list):
  w_list = []
  i = 0
  while i < len(_w_list):
    w = _w_list[i]
    if (w == 'the') and (i + 1 < len(_w_list)) and ((_w_list[i + 1] in month_set) or (word2num.has_key(_w_list[i + 1]))):
      pass
    else:
      w_list.append(w)
    i += 1

  _w_list = w_list
  w_list = []
  i = 0
  while i < len(_w_list):
    w = _w_list[i]
    if (i - 1 >= 0) and ((_w_list[i - 1] in month_set) or (_w_list[i - 1] in week_set) or (_w_list[i - 1] in ('on', 'and', 'for', 'early', 'late'))):
      if (i + 1 < len(_w_list)) and (_w_list[i + 1] == 'class'):
        w_list.append(w)
      elif (i + 1 < len(_w_list)) and word2num.has_key(''.join((w, _w_list[i + 1]))):
        w_list.append('_dn_' + word2num[''.join((w, _w_list[i + 1]))])
        i += 1
      elif word2num.has_key(w):
        w_list.append('_dn_' + word2num[w])
      else:
        w_list.append(w)
    elif ((i + 2 < len(_w_list)) and (_w_list[i + 1] == 'of') and (_w_list[i + 2] in month_set)):
      if (i - 1 >= 0) and word2num.has_key(''.join((_w_list[i - 1], w))):
        w_list[len(w_list) - 1] = '_dn_' + \
            word2num[''.join((_w_list[i - 1], w))]
      elif word2num.has_key(w):
        w_list.append('_dn_' + word2num[w])
      else:
        w_list.append(w)
    else:
      w_list.append(w)
    i += 1

  return w_list


def is_normalized_time_mention_str(m):
  if (len(m)==6) and (m.endswith('am') or m.endswith('pm')) and is_time_digit(m[:4]):
    return True
  return False
