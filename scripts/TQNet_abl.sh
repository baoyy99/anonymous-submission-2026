export CUDA_VISIBLE_DEVICES=0
#!/bin/bash
# loss="ema_tqa_mse""tqa_mae"

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
lradj=('type3' 'type3' 'type3' 'type3' 'sigmoid' 'sigmoid' 'sigmoid' 'sigmoid')
cycle=(24 24 96 96 168 144 144 168)
use_revin=(1 1 1 1 1 1 0 1)

for i in 4
do
#   for pred_len in 192 336 720
  for pred_len in 96
  do
    # for loss in "ema_tqa_mse" "tqa_mae"
    for loss in "tqa_mae"
    do
                echo "----- ${dataset_name[$i]} ${pred_len} -----"
                python -u 10-TQNet-master/run.py \
                    --is_training 1 \
                    --root_path ${root_path[$i]} \
                    --data_path ${data_path[$i]} \
                    --model TQNet \
                    --data ${data_type[$i]} \
                    --features $features \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --enc_in ${enc_in[$i]} \
                    --des 'Exp' \
                    --itr 1 \
                    --batch_size ${batch_size[$i]} \
                    --learning_rate ${learning_rate[$i]} \
                    --cycle ${cycle[$i]} \
                    --train_epochs ${train_epochs} \
                    --patience ${patience} \
                    --use_revin ${use_revin[$i]} \
                    --loss $loss\
                    --TQALoss_alpha 0\
                    --TQALoss_beta 1\
                    --TQALoss_gamma 0.1\
                    --TQALoss_q 0.6
           
    done
done
done
# done
# done
#  --TQALoss_q $TQALoss_q