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
beam_size=5
lstm='lstm'
model_name=model.jobs.sup.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.beam${beam_size}.${vocab}.${train_file}.no_par_prod_embed

python -u exp.py \
    --cuda \
    --mode train \
    --lang prolog \
    --no_parent_production_embed \
    --batch_size 10 \
    --asdl_file asdl/lang/prolog/prolog_asdl.txt \
    --train_file data/jobs/${train_file} \
    --dev_file data/jobs/test.bin \
    --vocab data/jobs/${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 10 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --decode_max_time_step 55 \
    --log_every 50 \
    --save_to saved_models/jobs/${model_name} 2>logs/jobs/${model_name}.log

. scripts/jobs/test.sh saved_models/jobs/${model_name}.bin 2>>logs/jobs/${model_name}.log


    # --no_parent_field_embed \
    # --no_parent_field_type_embed \
    # --no_parent_state \