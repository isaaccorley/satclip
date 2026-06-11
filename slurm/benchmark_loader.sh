#!/bin/bash
#SBATCH --job-name=bench_loader
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/bench_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/bench_%j.err

set -uo pipefail
export BENCH_DIR=/tmp/bench_$SLURM_JOB_ID   # node-local SSD -> isolates decode from NFS
export BENCH_N=2500
/projects/bgtj/isaaccorley/satclip-env/bin/python \
  /u/isaaccorley/github/satclip/scripts/benchmark_loader.py
rm -rf "$BENCH_DIR"
