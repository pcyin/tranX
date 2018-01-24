#!/bin/bash

train_data="train.5000.bin"
unlabeled_data="train.5000.remaining.bin"
lr_decay=0.5
encoder="model.sup.lstm_with_dropout.hidden_size256.embed128.action_embed128.field_embed64.type_embed64.dropout0.2.ptr_hidden32.lr_decay0.5.beam_size15.vocab.freq5.bin.train.5000.bin.bin"
decoder="model.sup.decoder.lstm.hidden_size256.embed128.dropout0.2.ptr_hidden32.lr_decay0.5.vocab.freq5.bin.train.5000.bin.bin"
unsup_loss_weight=0.1
model_name=model.semisup.${encoder}.unlabeled_${unlabeled_data}.unsup_loss_weight_${unsup_loss_weight}

python exp.py \
    --cuda \
    --mode train_semi \
    --batch_size 10 \
    --train_file ../data/django/${train_data} \
    --unlabeled_file ../data/django/${unlabeled_data} \
    --dev_file ../data/django/dev.bin \
    --load_model saved_models/${encoder} \
    --load_decoder saved_models/${decoder} \
    --unsup_loss_weight 0.1 \
    --patience 5 \
    --max_num_trial 3 \
    --lr_decay ${lr_decay} \
    --save_to saved_models/${model_name} 2>logs/${model_name}.log
