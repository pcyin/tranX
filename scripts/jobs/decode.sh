#!/bin/bash

python exp.py \
    --cuda \
    --mode test \
    --load_model $1 \
    --beam_size 5 \
    --test_file $2 \
    --save_decode_to decodes/$(basename $1).$(basename $2).decode \
    --decode_max_time_step 110
