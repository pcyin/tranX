# coding=utf-8
from __future__ import print_function

import sys, os, cPickle as pickle
import numpy as np
np.random.seed(0)

from exp import *

if __name__ == '__main__':
    full_train_file = sys.argv[1]
    full_train_set = pickle.load(open(full_train_file))

    splits = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 14000]
    np.random.shuffle(full_train_set)

    fname, ext = os.path.splitext(full_train_file)

    for split in splits:
        sup_examples = full_train_set[:split]
        remaining_examples = full_train_set[split:]

        pickle.dump(sup_examples, open(fname + '.%d' % split + ext, 'wb'))
        pickle.dump(remaining_examples, open(fname + '.%d.remaining' % split + ext, 'wb'))
