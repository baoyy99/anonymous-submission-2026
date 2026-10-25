#!/usr/bin/env bash

# RQ3: aligned with scripts/TimeFilter/TimeFilter_RQ3.sh.
# 4 datasets x 4 horizons x 3 seeds x (MSE + MAE + 4 TAQ modes at q=0.8) = 288 runs.
# RQ3_DRY_RUN=1 bash scripts/TQNet/CrossLinear_RQ3.sh prints commands only.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
trap 'exit 130' INT
trap 'exit 143' TERM

evaluation_config="TQNet/evaluation/configs/default.json"
if [[ ! -f TQNet/run.py || ! -f "${evaluation_config}" ]]; then
    echo "Run this script from /23-TAQLoss." >&2
    exit 2
fi

seq_len=96
pred_len_list=(720)
seed_list=(1998)
loss_gamma_list=(0.8)
features="M"
train_epochs=10
patience=3

# Learnable MODEL coefficients: unrelated to the loss_alpha/loss_beta loops.
model_alpha=1.0
model_beta=1.0
patch_len=16

dataset_name=("ETTh1" "Weather" "Electricity" "Traffic")
root_path=("./dataset/ETT-small" "./dataset/weather" "./dataset/electricity" "./dataset/traffic")
data_path=("ETTh1.csv" "weather.csv" "electricity.csv" "traffic.csv")
data_type=("ETTh1" "custom" "custom" "custom")
enc_in=(7 21 321 862)
dec_in=(7 21 321 862)
c_out=(7 21 321 862)
batch_size=(32 32 16 8)
learning_rate=(0.0001 0.0001 0.0001 0.0001)
# Explicit, modest starting configuration; fixed across objectives, not a tuned optimum.
d_model=(128 128 128 128)
d_ff=(128 128 128 128)

for i in 0 1 2 3
do
    if [[ ! -f "${root_path[$i]}/${data_path[$i]}" ]]; then
        echo "Missing dataset: ${root_path[$i]}/${data_path[$i]}" >&2
        exit 2
    fi
done

run_count=0
failure_count=0
total_runs=$((${#dataset_name[@]} * ${#pred_len_list[@]} * ${#seed_list[@]} * (2 + 4 * ${#loss_gamma_list[@]})))

run_one() {
    local i="$1"
    local loss="$2"
    local loss_alpha="$3"
    local loss_beta="$4"
    local loss_gamma="$5"
    local status
    local -a command

    run_count=$((run_count + 1))
    echo "[${run_count}/${total_runs}] CrossLinear ${dataset_name[$i]} pred_len=${pred_len} seed=${random_seed} loss=${loss} alpha=${loss_alpha} beta=${loss_beta} gamma=${loss_gamma}"

    command=(
        python -u TQNet/run.py
        --task_name long_term_forecast
        --is_training 1
        --random_seed "${random_seed}"
        --root_path "${root_path[$i]}"
        --data_path "${data_path[$i]}"
        --model CrossLinear
        --data "${data_type[$i]}"
        --features "${features}"
        --target OT
        --freq h
        --embed timeF
        --seq_len "${seq_len}"
        --label_len 0
        --pred_len "${pred_len}"
        --enc_in "${enc_in[$i]}"
        --dec_in "${dec_in[$i]}"
        --c_out "${c_out[$i]}"
        --patch_len "${patch_len}"
        --d_model "${d_model[$i]}"
        --d_ff "${d_ff[$i]}"
        --alpha "${model_alpha}"
        --beta "${model_beta}"
        --des Exp
        --itr 1
        --batch_size "${batch_size[$i]}"
        --learning_rate "${learning_rate[$i]}"
        --train_epochs "${train_epochs}"
        --patience "${patience}"
        --num_workers 0
        --loss "${loss}"
        --loss_alpha "${loss_alpha}"
        --loss_beta "${loss_beta}"
        --loss_gamma "${loss_gamma}"
        --evaluation_enabled
        --evaluation_config "${evaluation_config}"
        --evaluation_experiment_id RQ3
    )

    printf 'COMMAND:'
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "${RQ3_DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi

    if "${command[@]}"; then
        echo "[$(date '+%F %T')] 完成"
        return 0
    else
        status=$?
        if (( status == 130 || status == 143 )); then
            exit "${status}"
        fi
        failure_count=$((failure_count + 1))
        echo "FAILED (${status}): CrossLinear ${dataset_name[$i]} loss=${loss} alpha=${loss_alpha} beta=${loss_beta} gamma=${loss_gamma}" >&2
    fi
}

for random_seed in "${seed_list[@]}"
do
    for i in 3
    do
        for pred_len in "${pred_len_list[@]}"
        do
            # run_one "$i" mse 0 0 0
            # run_one "$i" mae 0 0 0
            for loss_alpha in 0 1
            do
                for loss_beta in 0 1
                do
                    for loss_gamma in "${loss_gamma_list[@]}"
                    do
                        run_one "$i" taq "${loss_alpha}" "${loss_beta}" "${loss_gamma}"
                    done
                done
            done
        done
    done
done

if [[ "${RQ3_DRY_RUN:-0}" == "1" ]]; then
    echo "RQ3 CrossLinear dry run: ${run_count} commands; no training started."
else
    echo "RQ3 CrossLinear finished: ${run_count} attempted, ${failure_count} failed."
fi
if (( failure_count > 0 )); then
    exit 1
fi
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command '1..5 | ForEach-Object { [Console]::Beep(1000,500); Start-Sleep -Milliseconds 500 }'