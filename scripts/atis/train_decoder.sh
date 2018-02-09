#!/bin/bash

vocab="vocab.freq10.bin"
train_file="train.1000.bin"
dropout=0.2
hidden_size=256
embed_size=128
ptrnet_hidden_dim=32
lr_decay=0.5
lstm='lstm'
model_name=model.sup.decoder.${lstm}.hidden${hidden_size}.embed${embed_size}.dropout${dropout}.lr_decay${lr_decay}.${vocab}.${train_file}

python exp.py \
    --cuda \
    --mode train_decoder \
    --batch_size 10 \
    --train_file ../data/django/${train_file} \
    --dev_file ../data/django/dev.bin \
    --vocab ../data/django/${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --ptrnet_hidden_dim ${ptrnet_hidden_dim} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --log_every 50 \
    --save_to saved_models/${model_name} 2>logs/${model_name}.log
