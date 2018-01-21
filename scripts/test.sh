#!/bin/bash

model_name=$(basename $1)

python exp.py \
	--cuda \
    --mode test \
    --load_model saved_models/${model_name} \
    --beam_size 15 \
    --test_file data/django/test.bin \
    --decode_max_time_step 100
