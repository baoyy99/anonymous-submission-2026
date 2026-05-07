#!/bin/bash
# nohup bash scripts/run_all.sh &

echo "==================== 1 ===================="
bash scripts/iTransformer.sh


echo "==================== 2 ===================="
bash scripts/CrossLinear.sh


echo "==================== 3 ===================="
bash scripts/TimeFilter.sh

echo "==================== done ===================="