#!/bin/bash

vocab="vocab.freq5.bin"
labeled_data="train.3000.bin"
unlabeled_file="train.3000.remaining.bin"
seed=1
decode_results="decodes/model.atis.sup.lstm.hidden256.embed128.action128.field32.type32.dropout0.3.lr_decay0.5.beam5.vocab.bin.train.3000.bin.bin.train.3000.remaining.bin.decode"
pretrained_model_name="saved_models/model.atis.sup.lstm.hidden256.embed128.action128.field32.type32.dropout0.3.lr_decay0.5.beam5.vocab.bin.train.3000.bin.bin"
model_name=self_train.$(basename ${pretrained_model_name}).${unlabeled_file}.seed${seed}

python exp.py \
    --cuda \
    --seed ${seed} \
    --mode self_train \
    --batch_size 10 \
    --train_file ../data/atis/${labeled_data} \
    --unlabeled_file ../data/atis/${unlabeled_file} \
    --load_decode_results ${decode_results} \
    --dev_file ../data/atis/dev.bin \
    --load_model ${pretrained_model_name} \
    --patience 5 \
    --max_num_trial 5 \
    --log_every 50 \
    --save_to saved_models/atis/${model_name} 2>logs/${model_name}.log

python exp.py \
	--cuda \
    --mode test \
    --load_model saved_models/atis/${model_name}.bin \
    --beam_size 5 \
    --test_file ../data/atis/test.bin \
    --decode_max_time_step 110 2>>logs/${model_name}.log
