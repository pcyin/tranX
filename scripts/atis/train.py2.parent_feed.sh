#!/bin/bash

vocab="vocab.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
lstm='parent_feed'
model_name=model.atis.sup.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.beam${beam_size}.${vocab}.${train_file}.py2

python -u exp.py \
    --cuda \
    --mode train \
    --lang lambda_dcs \
    --batch_size 10 \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --train_file data/atis/${train_file} \
    --dev_file data/atis/dev.bin \
    --vocab data/atis/${vocab} \
    --lstm ${lstm} \
    --no_parent_production_embed \
    --no_parent_field_embed \
    --no_parent_field_type_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/atis/${model_name} 2>logs/atis/${model_name}.log

. scripts/atis/test.sh saved_models/atis/${model_name}.bin 2>>logs/atis/${model_name}.log
