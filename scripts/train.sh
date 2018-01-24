#!/bin/bash

vocab="vocab.freq5.bin"
train_file="train.3000.bin"
dropout=0.2
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr_decay=0.5
beam_size=15
lstm='lstm_with_dropout'
model_name=model.sup.${lstm}.hidden_size${hidden_size}.embed${embed_size}.action_embed${action_embed_size}.field_embed${field_embed_size}.type_embed${type_embed_size}.dropout${dropout}.ptr_hidden${ptrnet_hidden_dim}.lr_decay${lr_decay}.beam_size${beam_size}.${vocab}.${train_file}

python exp.py \
    --cuda \
    --mode train \
    --batch_size 10 \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --train_file ../data/django/${train_file} \
    --dev_file ../data/django/dev.bin \
    --vocab ../data/django/${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --ptrnet_hidden_dim ${ptrnet_hidden_dim} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/${model_name} 2>logs/${model_name}.log

. scripts/test.sh saved_models/${model_name}.bin 2>>logs/${model_name}.log
