#!/bin/bash
#SBATCH --job-name=s2_distances
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/distances_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/distances_%j.err

set -euo pipefail
source /projects/bgtj/isaaccorley/satclip-env/bin/activate
cd /u/isaaccorley/github/satclip
python scripts/analyze_distances.py --n-jobs 32 --chunk 2000
