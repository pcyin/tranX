#!/bin/bash

train_data="train.1000.bin"
dropout=0.2
model_name=prior.${train_data}.dropout${dropout}

PYTHONPATH=. python scripts/atis/train_lstm_prior.py \
    --cuda \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --batch_size 10 \
    --train_file data/atis/${train_data} \
    --dev_file data/atis/dev.bin \
    --vocab data/atis/vocab.bin \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay 0.5 \
    --log_every 10 \
    --save_to saved_models/prior/${model_name} 2>logs/${model_name}.log
