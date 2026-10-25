#!/usr/bin/env bash

# RQ2 fixed-budget TQNet grid: 4 datasets x 16 objectives = 64 runs.


set -u
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f TQNet/run.py ]]; then
    echo "Run this script from /23-TAQLoss." >&2
    exit 2
fi

seq_len=96
pred_len=96
features="M"
train_epochs=10
patience=10
random_seed=1998

dataset_name=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" "weather" "solar" "electricity" "exchange_rate")
root_path=("./dataset/ETT-small" "./dataset/ETT-small" "./dataset/ETT-small" "./dataset/ETT-small" "./dataset/traffic" "./dataset/weather" "./dataset/Solar" "./dataset/electricity" "./dataset/exchange_rate")
data_path=("ETTh1.csv" "ETTh2.csv" "ETTm1.csv" "ETTm2.csv" "traffic.csv" "weather.csv" "solar_AL.txt" "electricity.csv" "exchange_rate.csv")
data_type=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "custom" "custom" "Solar" "custom" "custom")
enc_in=(7 7 7 7 862 21 137 321 8)
batch_size=(32 32 32 32 16 32 32 32 32)
learning_rate=(0.0001 0.0001 0.0001 0.0001 0.001 0.0001 0.0001 0.0001 0.0001)
lradj=("type3" "type3" "type3" "type3" "sigmoid" "sigmoid" "sigmoid" "sigmoid" "type1")
cycle=(24 24 96 96 168 144 144 168 24)
use_revin=(1 1 1 1 1 1 0 1 1)

run_count=0
failure_count=0

run_one() {
    local i="$1"
    local loss="$2"
    local alpha="$3"
    local beta="$4"
    local gamma="$5"
    run_count=$((run_count + 1))
    echo "[${run_count}/64] ${dataset_name[$i]} loss=${loss} alpha=${alpha} beta=${beta} gamma=${gamma}"

    if [[ "${RQ2_DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi

    if ! python -u TQNet/run.py \
        --is_training 1 \
        --random_seed "${random_seed}" \
        --root_path "${root_path[$i]}" \
        --data_path "${data_path[$i]}" \
        --model TQNet \
        --data "${data_type[$i]}" \
        --features "${features}" \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --enc_in "${enc_in[$i]}" \
        --des Exp \
        --itr 1 \
        --batch_size "${batch_size[$i]}" \
        --learning_rate "${learning_rate[$i]}" \
        --cycle "${cycle[$i]}" \
        --train_epochs "${train_epochs}" \
        --patience "${patience}" \
        --use_revin "${use_revin[$i]}" \
        --loss "${loss}" \
        --loss_alpha "${alpha}" \
        --loss_beta "${beta}" \
        --loss_gamma "${gamma}" \
        --evaluation_enabled \
        --evaluation_config TQNet/evaluation/configs/rq2_diagnostics.json \
        --evaluation_experiment_id RQ2
    then
        failure_count=$((failure_count + 1))
        echo "FAILED: ${dataset_name[$i]} loss=${loss} alpha=${alpha} beta=${beta} gamma=${gamma}" >&2
    fi
}

# ETTh1, Weather, Electricity, Traffic. The complete RQ2 grid adds the
# corresponding 64 iTransformer runs later; this script intentionally does not.
for i in 0 5 7 4
do
    run_one "$i" mse 0 0 0
    run_one "$i" mae 0 0 0

    for alpha in 0.2 0.8
    do
        run_one "$i" quantile "$alpha" 0 0
        run_one "$i" expectile "$alpha" 0 0
    done

    run_one "$i" slinex 0.8 0 0
    run_one "$i" ahuber 0.8 0 0

    for alpha in 0 1
    do
        for beta in 0 1
        do
            for gamma in 0.6 0.8
            do
                run_one "$i" taq "$alpha" "$beta" "$gamma"
            done
        done
    done
done

echo "RQ2 TQNet grid finished: ${run_count} scheduled, ${failure_count} failed."
if (( failure_count > 0 )); then
    exit 1
fi
