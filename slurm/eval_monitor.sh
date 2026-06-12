#!/bin/bash
#SBATCH --job-name=eval_monitor
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=18:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/evalmon_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/evalmon_%j.err
source /projects/bgtj/isaaccorley/satclip-env/bin/activate
python /u/isaaccorley/github/satclip/scripts/eval_monitor.py
