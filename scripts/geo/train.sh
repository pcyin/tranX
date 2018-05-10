#!/bin/bash

seed=${1:-0}
vocab="vocab.freq2.bin"
train_file="train.bin"
dropout=0.5
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=200
patience=1000   # disable patience since we don't have dev set
beam_size=5
batch_size=10
lr=0.002
lstm='lstm'
model_name=model.geo.sup.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.field${field_embed_size}.type${type_embed_size}.drop${dropout}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.${vocab}.${train_file}.pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.par_state_w_field_embed.seed${seed}

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --lang lambda_dcs \
    --batch_size ${batch_size} \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --train_file data/geo/${train_file} \
    --vocab data/geo/${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_all_models \
    --save_to saved_models/geo/${model_name}/model 2>logs/geo/${model_name}.log

    # --sup_attention \
    # --no_parent_production_embed \
    # --no_parent_field_embed \
    # --no_parent_field_type_embed \
    # --no_parent_state \

# python exp.py \
#     --cuda \
#     --mode test \
#     --load_model saved_models/geo/${model_name}.bin \
#     --beam_size 5 \
#     --test_file data/geo/dev.bin \
#     --decode_max_time_step 110 2>>logs/geo/${model_name}.log

. scripts/geo/test.sh saved_models/geo/${model_name}/model.bin 2>>logs/geo/${model_name}.log
