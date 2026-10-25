#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

seq_len=96
label_len=48
features=M
train_epochs=10
patience=3
pred_len_list=(96 192 336 720)
seed_list=(1998 2024 2025)

dataset_name=("ETTh1" "Weather" "Electricity" "Traffic")
root_path=("./dataset/ETT-small" "./dataset/weather" "./dataset/electricity" "./dataset/traffic")
data_path=("ETTh1.csv" "weather.csv" "electricity.csv" "traffic.csv")
data_type=("ETTh1" "custom" "custom" "custom")
enc_in=(7 21 321 862)
batch_size=(32 32 32 16)
learning_rate=(0.0001 0.0001 0.0001 0.001)
patch_len=(2 48 32 96)
pos=(0 1 1 0)
d_model=(128 128 512 512)
d_ff=(256 256 512 2048)

run_count=0
for seed in "${seed_list[@]}"
do
for i in {0..3}
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
echo "[TimeFilter ${run_count}/288] ${dataset_name[$i]} pred_len=${pred_len} seed=${seed} loss=${loss} alpha=${loss_alpha} beta=${loss_beta} gamma=${loss_gamma}"
python -u TimeFilter/run.py \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id P3_${dataset_name[$i]} \
    --root_path ${root_path[$i]} \
    --data_path ${data_path[$i]} \
    --model TimeFilter \
    --data ${data_type[$i]} \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${enc_in[$i]} \
    --dec_in ${enc_in[$i]} \
    --c_out ${enc_in[$i]} \
    --patch_len ${patch_len[$i]} \
    --pos ${pos[$i]} \
    --d_model ${d_model[$i]} \
    --d_ff ${d_ff[$i]} \
    --des P3 \
    --itr 1 \
    --batch_size ${batch_size[$i]} \
    --learning_rate ${learning_rate[$i]} \
    --train_epochs $train_epochs \
    --patience $patience \
    --seed $seed \
    --loss $loss \
    --loss_alpha $loss_alpha \
    --loss_beta $loss_beta \
    --loss_gamma $loss_gamma \
    --loss_q 0.5 \
    --evaluation_enabled \
    --evaluation_config TimeFilter/evaluation/configs/rq3_metrics_only.json \
    --evaluation_root TimeFilter/evaluation \
    --evaluation_experiment_id RQ3
done
done
done
done
done
done
done
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command '1..5 | ForEach-Object { [Console]::Beep(1000,500); Start-Sleep -Milliseconds 500 }'

