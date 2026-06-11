#!/bin/bash
#SBATCH --job-name=bench_dtype
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/benchdt_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/benchdt_%j.err
set -uo pipefail
export BENCH_DIR=/tmp/bench_dtype_$SLURM_JOB_ID
export BENCH_N=2500
/projects/bgtj/isaaccorley/satclip-env/bin/python /u/isaaccorley/github/satclip/scripts/bench_dtype.py
rm -rf "$BENCH_DIR"
