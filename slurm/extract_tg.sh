#!/bin/bash
#SBATCH --job-name=tg_extract
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/tg_extract_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/tg_extract_%j.err

set -uo pipefail
ROOT=/projects/bgtj/isaaccorley/s2-100k-tg
cd "$ROOT"

# satclip.tar is a single UNCOMPRESSED tar of flat patch_*.tif -> extract into
# images/ so the data module sees ROOT/index.csv + ROOT/images/patch_*.tif.
mkdir -p images
tar -xf satclip.tar -C images/
rc=$?
echo "tar exit=$rc"

count=$(ls images/patch_*.tif 2>/dev/null | wc -l)
echo "extracted patch count: $count"
[ -f images/patch_99999.tif ] && echo "patch_99999.tif OK" || { echo "patch_99999 MISSING"; rc=1; }
if [ "$count" -ne 100000 ] || [ "$rc" -ne 0 ]; then
  echo "EXTRACT INCOMPLETE (count=$count, rc=$rc)"; exit 1
fi

# Authoritative integrity gate: open+read every tif, delete any corrupt.
/projects/bgtj/isaaccorley/satclip-env/bin/python \
  /u/isaaccorley/github/satclip/scripts/validate_patches.py \
  --images "$ROOT/images" --n-jobs 16 --delete
echo "valid patch count: $(ls images/patch_*.tif 2>/dev/null | wc -l)"
echo "EXTRACT DONE"
