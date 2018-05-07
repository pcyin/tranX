#!/bin/bash

source activate python3

vocab="vocab.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
embed_size=100
action_embed_size=100
field_embed_size=32
type_embed_size=32
lr_decay=0.7
beam_size=5
lstm='lstm'
model_name=model.wikisql.sup.exe_acc.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.beam${beam_size}.${vocab}.${train_file}.glorot.par_state_w_field_embed

echo commit hash: `git rev-parse HEAD` > logs/wikisql/${model_name}.log

python -u exp.py \
    --cuda \
    --mode train \
    --lang wikisql \
    --batch_size 64 \
    --asdl_file asdl/lang/sql/sql_asdl.txt \
    --train_file data/wikisql/${train_file} \
    --dev_file data/wikisql/dev.bin \
    --vocab data/wikisql/${vocab} \
    --glove_embed_path ../datasets/glove.6B/glove.6B.100d.txt \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 50 \
    --log_every 10 \
    --save_to saved_models/wikisql/${model_name} 2>>logs/wikisql/${model_name}.log

. scripts/wikisql/test.sh saved_models/wikisql/${model_name}.bin 2>>logs/wikisql/${model_name}.log
