#!/bin/bash

vocab="vocab.bin"
train_file="train.500.bin"
dropout=0.3
hidden_size=256
embed_size=128
lr_decay=0.5
lstm='lstm'
model_name=model.atis.sup.decoder.${lstm}.hidden${hidden_size}.embed${embed_size}.dropout${dropout}.lr_decay${lr_decay}.${vocab}.${train_file}

python exp.py \
    --cuda \
    --lang lambda_dcs \
    --asdl_file lang/lambda_dcs/lambda_asdl.txt \
    --mode train_decoder \
    --batch_size 10 \
    --train_file ../data/atis/${train_file} \
    --dev_file ../data/atis/dev.bin \
    --vocab ../data/atis/${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --log_every 50 \
    --save_to saved_models/${model_name} 2>logs/${model_name}.log
