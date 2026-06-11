#!/bin/bash
#SBATCH --job-name=satclip
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/%x_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/%x_%j.err

# Usage: sbatch --job-name=satclip_baseline slurm/train.sh configs/baseline.yaml
set -euo pipefail

CONFIG="${1:?Pass a config path relative to satclip/, e.g. configs/baseline.yaml}"

source /projects/bgtj/isaaccorley/satclip-env/bin/activate

cd /u/isaaccorley/github/satclip/satclip

export SATCLIP_CONFIG="./${CONFIG}"
export WANDB_DIR="/projects/bgtj/isaaccorley/satclip_runs"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "config=$SATCLIP_CONFIG"
nvidia-smi --query-gpu=memory.total --format=csv,noheader

python main.py
