#!/bin/bash

train_data="full"
unlabeled_data=""
lr_decay=0.5
encoder=""
decoder=""
model_name=model.semisup.encoder${encoder}.decoder${decoder}.lr_decay${lr_decay}.label_data${train_data}.unlabeled_data${unlabeled_data}

python exp.py \
    --cuda \
    --mode train \
    --batch_size 10 \
    --train_file data/django/${train_data} \
    --unlabeled_file data/django/${unlabeled_data} \
    --dev_file data/django/dev.bin \
    --load_model saved_models/${encoder} \
    --load_decoder saved_models/${decoder} \
    --patience 5 \
    --max_num_trial 3 \
    --lr_decay ${lr_decay} \
    --save_to saved_models/${model_name} 2>logs/${model_name}.log
