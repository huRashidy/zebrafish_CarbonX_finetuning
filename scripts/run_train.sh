#!/usr/bin/env bash
set -e

cd /home/hussin/zebrafish_ft
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

echo "=================================================="
echo "Starting Full Carbon-X QLoRA Training on Zebrafish"
echo "Date: $(date)"
echo "Config: configs/qlora.yaml"
echo "=================================================="

/home/hussin/miniconda3/envs/benchmark_env/bin/python src/train.py --config configs/qlora.yaml "$@"
