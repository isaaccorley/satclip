#!/bin/bash
#SBATCH --job-name=zstd_reencode
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/zstd_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/zstd_%j.err

set -uo pipefail
export SRC=/projects/bgtj/isaaccorley/s2-100k-tg
export DST=/projects/bgtj/isaaccorley/s2-100k-tg-zstd
export N_JOBS=64
PY=/projects/bgtj/isaaccorley/satclip-env/bin/python

$PY /u/isaaccorley/github/satclip/scripts/reencode_zstd.py || exit 1

# Authoritative integrity gate: open+read every re-encoded tif.
$PY /u/isaaccorley/github/satclip/scripts/validate_patches.py \
  --images "$DST/images" --n-jobs 64 --delete

echo "final valid count: $(ls $DST/images/patch_*.tif 2>/dev/null | wc -l)"
echo "disk usage: $(du -sh $DST/images 2>/dev/null | cut -f1)"
echo "ZSTD REENCODE DONE"
