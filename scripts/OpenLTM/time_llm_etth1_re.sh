#!/bin/bash

# 从仓库根目录运行：bash scripts/OpenLTM/time_llm_etth1_re.sh
# 只重跑历史早停核验中的 8 组；任一组失败后停止，避免继续提交。
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

model_name=time_llm
token_len=96
seq_len=96

dataset_name=("ETTh1")
root_path=("./dataset/ETT-small")
data_path=("ETTh1.csv")
batch_size=(4)
i=0

# 每列对应同一组实验，顺序与核验名单一致。
# 历史“应停止轮次”仅供核验，不用于硬编码训练轮数。
pred_len_list=(336 192 192 720 96 336 336 336)
loss_list=(mae mse taq taq taq taq taq taq)
loss_alpha_list=(0 0 0 0 0 0 1 1)
loss_beta_list=(0 0 0 0 1 1 0 1)
loss_gamma_list=(0 0 0.6 0.8 0.6 0.6 0.6 0.6)

run_count=0
for j in "${!pred_len_list[@]}"
do
pred_len=${pred_len_list[$j]}
loss=${loss_list[$j]}
loss_alpha=${loss_alpha_list[$j]}
loss_beta=${loss_beta_list[$j]}
loss_gamma=${loss_gamma_list[$j]}

run_count=$((run_count + 1))
echo "[Time-LLM rerun ${run_count}/8] ${dataset_name[$i]} pred_len=${pred_len} loss=${loss} alpha=${loss_alpha} beta=${loss_beta} gamma=${loss_gamma}"
# 打印展开后的完整启动命令，便于检查日志。
set -x
python -u OpenLTM/run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path "${root_path[$i]}" \
    --data_path "${data_path[$i]}" \
    --model_id "P3_${dataset_name[$i]}" \
    --model "$model_name" \
    --data MultivariateDatasetBenchmark \
    --seq_len "$seq_len" \
    --input_token_len "$token_len" \
    --output_token_len "$pred_len" \
    --test_seq_len "$seq_len" \
    --test_pred_len "$pred_len" \
    --d_model 768 \
    --d_ff 768 \
    --batch_size "${batch_size[$i]}" \
    --learning_rate 1e-5 \
    --cosine \
    --train_epochs 10 \
    --patience 3 \
    --use_norm \
    --pretrain_model_path OpenLTM/pretrained/gpt2 \
    --checkpoints OpenLTM/checkpoint \
    --gpu 0 \
    --seed 1998 \
    --llm_model GPT2 \
    --llm_layers 6 \
    --patch_size 16 \
    --stride 8 \
    --tmax 10 \
    --valid_last \
    --nonautoregressive \
    --subset_rand_ratio 1.0 \
    --loss "$loss" \
    --loss_alpha "$loss_alpha" \
    --loss_beta "$loss_beta" \
    --loss_gamma "$loss_gamma" \
    --des P3 \
    --evaluation_enabled \
    --evaluation_config OpenLTM/evaluation/configs/p3_model_comparison.json \
    --evaluation_root OpenLTM/evaluation \
    --evaluation_experiment_id P3
set +x
done

echo "Time-LLM ETTh1：8 组早停重跑已全部完成。"
