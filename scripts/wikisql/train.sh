#!/bin/bash
set -e

source activate py3torch3cuda9

seed=${1:-0}
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
model_name=model.wikisql.sup.exe_acc.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr_decay${lr_decay}.pat${patience}.beam${beam_size}.${vocab}.${train_file}.col_att_${col_att}.glorot.no_par_info.seed${seed}

echo commit hash: `git rev-parse HEAD` > logs/wikisql/${model_name}.log

echo `which python`

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 64 \
    --parser wikisql_parser \
    --asdl_file asdl/lang/sql/sql_asdl.txt \
    --transition_system sql \
    --evaluator wikisql_evaluator \
    --train_file data/wikisql/${train_file} \
    --dev_file data/wikisql/dev.bin \
    --sql_db_file data/wikisql/dev.db \
    --vocab data/wikisql/${vocab} \
    --glove_embed_path data/contrib/glove.6B.100d.txt \
    --lstm ${lstm} \
    --column_att ${col_att} \
    --no_parent_state \
    --no_parent_field_embed \
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
    --eval_top_pred_only \
    --decode_max_time_step 50 \
    --log_every 10 \
    --save_to saved_models/wikisql/${model_name} 2>>logs/wikisql/${model_name}.log

. scripts/wikisql/test.sh saved_models/wikisql/${model_name}.bin 2>>logs/wikisql/${model_name}.log
