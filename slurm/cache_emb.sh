#!/bin/bash
#SBATCH --job-name=cache_emb
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/cache_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/cache_%j.err
source /projects/bgtj/isaaccorley/satclip-env/bin/activate
python /u/isaaccorley/github/satclip/scripts/cache_image_embeddings.py
