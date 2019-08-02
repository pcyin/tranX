#!/bin/bash

model_name=$(basename $1)

python exp.py \
    --cuda \
    --mode test \
    --load_model $1 \
    --beam_size 5 \
    --parser wikisql_parser \
    --evaluator wikisql_evaluator \
    --sql_db_file data/wikisql/test.db \
    --test_file data/wikisql/test.bin \
    --save_decode_to decodes/wikisql/${model_name}.decode \
    --decode_max_time_step 50
