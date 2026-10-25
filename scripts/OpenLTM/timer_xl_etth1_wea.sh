#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=timer_xl
token_num=1
token_len=96
seq_len=$[$token_num*$token_len]
pred_len_list=(96)

dataset_name=("ETTh1" "Weather" "Electricity" "Traffic")
root_path=("./dataset/ETT-small" "./dataset/weather" "./dataset/electricity" "./dataset/traffic")
data_path=("ETTh1.csv" "weather.csv" "electricity.csv" "traffic.csv")
batch_size=(4 2 1 1)

run_count=0
for i in  1
do
for pred_len in "${pred_len_list[@]}"
do
for loss in "mse" "mae" "taq"
do
if [ "$loss" = "taq" ]; then
    loss_alpha_list=(0 1)
    loss_beta_list=(0 1)
    loss_gamma_list=(0.8)
else
    loss_alpha_list=(0)
    loss_beta_list=(0)
    loss_gamma_list=(0)
fi
for loss_alpha in "${loss_alpha_list[@]}"
do
for loss_beta in "${loss_beta_list[@]}"
do
for loss_gamma in "${loss_gamma_list[@]}"
do
run_count=$((run_count + 1))
echo "[Timer-XL ${run_count}/160] ${dataset_name[$i]} pred_len=${pred_len} loss=${loss} alpha=${loss_alpha} beta=${loss_beta} gamma=${loss_gamma}"
python -u OpenLTM/run.py \
    --task_name forecast \
    --is_training 1 \
    --root_path ${root_path[$i]} \
    --data_path ${data_path[$i]} \
    --model_id P3_${dataset_name[$i]} \
    --model $model_name \
    --data MultivariateDatasetBenchmark \
    --seq_len $seq_len \
    --input_token_len $token_len \
    --output_token_len $token_len \
    --test_seq_len $seq_len \
    --test_pred_len $pred_len \
    --e_layers 8 \
    --d_model 1024 \
    --d_ff 2048 \
    --batch_size ${batch_size[$i]} \
    --learning_rate 5e-6 \
    --train_epochs 10 \
    --patience 3 \
    --gpu 0 \
    --seed 1998 \
    --cosine \
    --tmax 10 \
    --use_norm \
    --adaptation \
    --pretrain_model_path OpenLTM/pretrained/timer_xl_260b/checkpoint.pth \
    --checkpoints OpenLTM/checkpoint \
    --subset_rand_ratio 1.0 \
    --loss $loss \
    --loss_alpha $loss_alpha \
    --loss_beta $loss_beta \
    --loss_gamma $loss_gamma \
    --des P3 \
    --evaluation_enabled \
    --evaluation_config OpenLTM/evaluation/configs/p3_model_comparison.json \
    --evaluation_root OpenLTM/evaluation \
    --evaluation_experiment_id P3
date '+ %Y-%m-%d %H:%M:%S'
done
done
done
done
done
done
