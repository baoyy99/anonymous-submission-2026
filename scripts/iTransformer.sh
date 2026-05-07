export CUDA_VISIBLE_DEVICES=0
#!/bin/bash
# loss="mse"
# loss="mae"
loss="pall_ema_tqa_mae"
# loss="pall_ema_tqa_mse"

seq_len=96
label_len=48
features="M"
embed="timeF"
target="OT"
freq="h"
train_epochs=10
patience=3


dataset_name=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" "weather" "solar"  "electricity")
root_path=("./dataset2026/ETT-small"
    "./dataset2026/ETT-small"
    "./dataset2026/ETT-small"
    "./dataset2026/ETT-small"
    "./dataset2026/traffic"
    "./dataset2026/weather"
    "./dataset2026/Solar"
    "./dataset2026/electricity"
)
data_path=("ETTh1.csv"
    "ETTh2.csv"
    "ETTm1.csv"
    "ETTm2.csv"
    "traffic.csv"
    "weather.csv"
    "solar_AL.txt"
    "electricity.csv"
)
data_type=("ETTh1"
    "ETTh2"
    "ETTm1"
    "ETTm2"
    "custom"
    "custom"
    "Solar"
    "custom"
)
enc_in=(7 7 7 7 862 21 137 321)
dec_in=(7 7 7 7 862 21 137 321)
c_out=(7 7 7 7 862 21 137 321)
batch_size=(32 32 32 32 16 32 32 32)
learning_rate=(0.0001 0.0001 0.0001 0.0001 0.001 0.0001 0.0001 0.0001)

e_layers=(2 2 2 2 4 3 3 3)
d_model=(128 128 128 128 512 512 512 512)
d_ff=(128 128 128 128 512 512 512 512)

# for i in {0..7}
for i in 0
do
#   for pred_len in 96 192 336 720
  for pred_len in 96
  do

        echo "----- ${dataset_name[$i]} ${pred_len} -----"
        python -u 12-tslib-asymmetry/run.py \
            --is_training 1 \
            --task_name long_term_forecast \
            --model_id $pred_len\
            --root_path ${root_path[$i]} \
            --data_path ${data_path[$i]} \
            --model iTransformer \
            --data ${data_type[$i]} \
            --features $features \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --e_layers ${e_layers[$i]} \
            --d_layers 1 \
            --factor 3 \
            --enc_in ${enc_in[$i]} \
            --dec_in ${dec_in[$i]} \
            --c_out ${c_out[$i]} \
            --d_model ${d_model[$i]} \
            --d_ff ${d_ff[$i]} \
            --des 'Exp' \
            --itr 1 \
            --batch_size ${batch_size[$i]} \
            --learning_rate ${learning_rate[$i]} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --loss $loss\
            --TQALoss_alpha 0\
            --TQALoss_beta 1\
            --TQALoss_gamma 0.1\
            --TQALoss_q 0.6
    done
done


