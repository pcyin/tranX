#!/bin/bash

source activate python3

seed=2
vocab="vocab.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
embed_size=100
action_embed_size=100
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
patience=5
lstm='lstm'
col_att='affine'
model_name=model.wikisql.sup.exe_acc.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.pat${patience}.beam${beam_size}.${vocab}.${train_file}.col_att_${col_att}.glorot.par_state_w_field_embed.seed${seed}

echo commit hash: `git rev-parse HEAD` > logs/wikisql/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --lang wikisql \
    --batch_size 64 \
    --asdl_file asdl/lang/sql/sql_asdl.txt \
    --train_file data/wikisql/${train_file} \
    --dev_file data/wikisql/dev.bin \
    --vocab data/wikisql/${vocab} \
    --glove_embed_path ../glove.6B/glove.6B.100d.txt \
    --lstm ${lstm} \
    --column_att ${col_att} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 50 \
    --log_every 10 \
    --save_to saved_models/wikisql/${model_name} 2>>logs/wikisql/${model_name}.log

. scripts/wikisql/test.sh saved_models/wikisql/${model_name}.bin 2>>logs/wikisql/${model_name}.log
