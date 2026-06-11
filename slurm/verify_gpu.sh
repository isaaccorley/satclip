#!/bin/bash
#SBATCH --job-name=verify_gpu
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/verifygpu_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/verifygpu_%j.err
/projects/bgtj/isaaccorley/satclip-env/bin/python /u/isaaccorley/github/satclip/scripts/verify_gpu_transform.py
