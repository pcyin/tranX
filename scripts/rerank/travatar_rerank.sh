#!/bin/bash
set -e

TDIR=$HOME/Research/travatar
work_dir=$1
suffix=$2
# seed=$1

mkdir -p ${work_dir}/model
mkdir -p ${work_dir}/logs

# split features
for f in dev test; do
  cp ${work_dir}/${f}${suffix}.hyp ${work_dir}/${f}${suffix}.hyp.all
  python scripts/rerank/split_feat.py ${work_dir}/${f}${suffix}.hyp
done

echo "parser_score=1 reconstructor=1 paraphrase_identifier=1" > weight_in.txt

# MERT + Reranking
for feat in all "+recon" "+para" "+recon+para" "+recon+wc" "+para+wc" "+recon+np" "+para+np" "+recon+para+wc" "+recon+para+np"; do
  train_hyp_file=dev${suffix}.hyp.${feat}
  save_model_name=${train_hyp_file}.weights
  echo process ${train_hyp_file}
  $TDIR/src/bin/batch-tune -nbest ${work_dir}/${train_hyp_file} -rand_seed 1 -algorithm lbfgs -l2 1 -weight_in weight_in.txt -eval zeroone -restarts 10 -threads 4 ${work_dir}/dev${suffix}.tgt > ${work_dir}/model/${save_model_name}

  for f in dev test; do
    hyp_file=${f}${suffix}.hyp.${feat}
    $TDIR/src/bin/rescorer -nbest ${work_dir}/${hyp_file} -weight_in ${work_dir}/model/${save_model_name} -nbest_out ${work_dir}/model/${hyp_file}.nbest > ${work_dir}/model/${hyp_file}.tgt
    $TDIR/src/bin/mt-evaluator -eval "bleu zeroone" -ref ${work_dir}/${f}${suffix}.tgt ${work_dir}/model/${hyp_file}.tgt > ${work_dir}/logs/${hyp_file}.log 2>&1
  done
done

