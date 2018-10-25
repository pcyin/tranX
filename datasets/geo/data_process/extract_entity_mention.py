# -*- coding: utf-8 -*-
import re

entity_re = re.compile(r' (\w+?):(\w+?) ')


def read_mention(fn='data/geoqueries/city_mention.txt'):
  m_dict = {}
  with open(fn, 'r') as f_in:
    l_list = filter(lambda x: (len(x) > 0) and (not x.startswith('//')), map(
        lambda x: x.strip(), f_in.read().decode('utf-8').split('\n')))
  for l in l_list:
    sp = l.split('\t')
    m_dict[sp[0]] = sp[1:]
  return m_dict

m_dict = read_mention()


def extract_main(split, e2m_dict):
  with open('data/geoqueries/%s.raw' % (split,), 'r') as f_in:
    raw_list = filter(lambda x: (len(x) > 0) and (not x.startswith('//')), map(
        lambda x: x.strip(), f_in.read().decode('utf-8').split('\n')))
    qf_list = [(raw_list[2 * i], raw_list[2 * i + 1])
               for i in xrange(len(raw_list) / 2)]
  for q, f in qf_list:
    q = ' %s ' % (q,)
    f = '  ' + ' '.join([it for it in filter(lambda x: len(x) > 0, f.replace(
        '(', ' ( ').replace(')', ' ) ').strip().split(' '))]).replace(' ', '  ') + '  '
    for e_name, e_type in entity_re.findall(f):
      k = '%s:%s' % (e_name, e_type)
      tk_list = e_name.split('_')
      if e_type in ('n', 's', 'r', 'c', 'm', 'co'):
        if e_type in ('m', 'co'):
          e2m_dict[k] = [' '.join(tk_list)]
        elif e_type == 's':
          s = ' '.join(tk_list)
          e2m_dict[k] = [s, '%s state'%(s,)]
        elif e_type == 'n':
          s = ' '.join(tk_list)
          e2m_dict[k] = ['named '+s, 'called '+s, 'a city of '+s]
        elif e_type == 'r':
          e2m_dict[k] = [' '.join(tk_list)]
        elif e_type == 'c':
          if e_name != 'capital':
            s = ' '.join(tk_list)
            m = ' '.join(tk_list[:-1] if len(tk_list) > 1 else tk_list)
            if m in ('new york', 'washington'):
              e2m_dict[k] = [s, '%s city'%(m,), 'the city of %s'%(m,)]
            else:
              e2m_dict[k] = [s, m, '%s city'%(m,), 'the city of %s'%(m,)]
        if m_dict.has_key(k):
          e2m_dict[k].extend(m_dict[k])
        found = False
        for m in e2m_dict.get(k, []):
          if q.find(' %s ' % (m,)) >= 0:
            found = True
            break
        if not found:
          if k != 'capital:c':
            print k, q, f
      # else:
      #   e2m_dict[k] = [k]

if __name__ == '__main__':
  e2m_dict = {}
  for split in ('train', 'test'):
    print split, ':'
    extract_main(split, e2m_dict)
  with open('data/geoqueries/entity_mention.txt', 'w') as f_out:
    for k, v in sorted([(k, v) for k, v in e2m_dict.iteritems()], key=lambda x: x[0].split(':')[1]):
      print >>f_out, ('%s\t%s' % (k, '\t'.join(
          sorted(list(set(v)), key=lambda x: len(x), reverse=True)))).encode('utf-8')
