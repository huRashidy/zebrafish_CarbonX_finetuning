#!/usr/bin/env bash
set -e

cd /home/hussin/zebrafish_ft
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

echo "=================================================="
echo "Evaluating Fine-Tuned Carbon-X on Zebrafish Val"
echo "Date: $(date)"
echo "Adapter: outputs/carbonx-zf-qlora"
echo "=================================================="

/home/hussin/miniconda3/envs/benchmark_env/bin/python src/evaluate.py \
    --config configs/qlora.yaml \
    --test-csv data/splits/val.csv \
    --output-csv results/val_predictions.csv \
    --adapter-path outputs/carbonx-zf-qlora "$@"
