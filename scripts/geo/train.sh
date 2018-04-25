#!/bin/bash

source activate python3

vocab="vocab.freq2.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.5
patience=10000
beam_size=5
lstm='lstm'
model_name=model.geo.sup.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.beam${beam_size}.${vocab}.${train_file}.patience${patience}

python -u exp.py \
    --cuda \
    --mode train \
    --lang lambda_dcs \
    --batch_size 10 \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --train_file data/geo/${train_file} \
    --dev_file data/geo/test.bin \
    --vocab data/geo/${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/geo/${model_name} 2>logs/geo/${model_name}.log

. scripts/geo/test.sh saved_models/geo/${model_name}.bin 2>>logs/geo/${model_name}.log
