#!/bin/bash

test_file="data/django/test.bin"

python exp.py \
	--cuda \
    --mode test \
    --load_model $1 \
    --beam_size 15 \
    --test_file ${test_file} \
    --save_decode_to decodes/django/$(basename $1).test.decode \
    --decode_max_time_step 100
