#!/bin/bash

# bash scripts/OpenLTM/run_all.sh

# echo "==================== [1/4] GPT4TS：160===================="
# bash scripts/OpenLTM/gpt4ts_etth1.sh

# echo "==================== [2/4] Time-LLM：160 ===================="
# bash scripts/OpenLTM/time_llm_etth1.sh

echo "==================== [3/4] TimesFM：160 ===================="
bash scripts/OpenLTM/timesfm_etth1.sh

echo "==================== [4/4] Timer-XL：160 ===================="
bash scripts/OpenLTM/timer_xl_etth1.sh

echo "==================== [1/4] GPT4TS  ===================="
bash scripts/OpenLTM/gpt4ts_etth1_seed2.sh

echo "==================== [2/4] Time-LLM  ===================="
bash scripts/OpenLTM/time_llm_etth1_seed2.sh

echo "==================== [3/4] TimesFM  ===================="
bash scripts/OpenLTM/timesfm_etth1_seed2.sh

echo "==================== [4/4] Timer-XL  ===================="
bash scripts/OpenLTM/timer_xl_etth1_seed2.sh

echo "==================== done ===================="
