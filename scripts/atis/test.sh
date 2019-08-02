#!/bin/bash

model_name=$(basename $1)

python exp.py \
    --mode test \
    --load_model $1 \
    --beam_size 5 \
    --test_file data/atis/test.bin \
    --save_decode_to decodes/atis/${model_name}.test.decode \
    --decode_max_time_step 110
