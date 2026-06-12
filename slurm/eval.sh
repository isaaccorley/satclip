#!/bin/bash
#SBATCH --job-name=satclip_eval
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/eval_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/eval_%j.err
source /projects/bgtj/isaaccorley/satclip-env/bin/activate
python /u/isaaccorley/github/satclip/scripts/eval_loc_encoder.py
