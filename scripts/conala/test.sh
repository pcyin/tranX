#!/bin/bash

source activate py3torch3cuda9

test_file="data/conala/test.var_str_sep.bin"

python exp.py \
    --cuda \
    --mode test \
    --load_model $1 \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --save_decode_to decodes/conala/$(basename $1).test.decode \
    --decode_max_time_step 100

