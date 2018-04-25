#!/bin/bash

vocab="vocab.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
embed_size=128
lr_decay=0.5
model_name=src_lm.atis.hidden${hidden_size}.embed${embed_size}.dropout${dropout}.lr_decay${lr_decay}.${vocab}.${train_file}

PYTHONPATH=. python scripts/train_source_lm.py \
    --cuda \
    --batch_size 10 \
    --train_file ../data/atis/${train_file} \
    --dev_file ../data/atis/dev.bin \
    --vocab ../data/atis/${vocab} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --log_every 50 \
    --save_to saved_models/${model_name} 2>logs/${model_name}.log
