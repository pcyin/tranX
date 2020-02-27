#!/bin/bash
set -e

seed=0
vocab="data/conala/vocab.src_freq3.code_freq3.mined_100000.snippet5.bin"
train_file="data/conala/train.var_str_sep.bin"
dev_file="data/conala/dev.bin"
train_decode_file="decodes/conala/finetune.mined.retapi.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.seed0.mined_100000.snippet5.bin.train.var_str_sep.bin.decode"
dev_decode_file="decodes/conala/finetune.mined.retapi.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.seed0.mined_100000.snippet5.bin.dev.bin.decode"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.0005
lr_decay=0.5
batch_size=10
max_epoch=80
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=paraphrase_identifier

echo "**** Writing results to logs/conala/${model_name}.log ****"
mkdir -p logs/conala
echo commit hash: `git rev-parse HEAD` > logs/conala/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train_paraphrase_identifier \
    --batch_size ${batch_size} \
    --evaluator conala_evaluator \
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --train_decode_file ${train_decode_file} \
    --dev_decode_file ${dev_decode_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --max_epoch ${max_epoch} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/conala/${model_name} 2>&1 | tee logs/conala/${model_name}.log
