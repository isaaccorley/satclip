#!/bin/bash
#SBATCH --job-name=s2_extract
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/bgtj/isaaccorley/satclip_runs/logs/extract_%j.out
#SBATCH --error=/projects/bgtj/isaaccorley/satclip_runs/logs/extract_%j.err

set -uo pipefail
ROOT=/projects/bgtj/isaaccorley/s2-100k
cd "$ROOT"

# Clean slate: drop any previously-extracted (possibly race-corrupted) patches.
rm -f images/patch_*.tif

# Extract each shard into its OWN staging dir in parallel. No two processes ever
# write the same path, so concurrent extraction cannot interleave/corrupt a file.
status_dir=$(mktemp -d)
rm -rf stage && mkdir -p stage
for f in images/data_*.tar.xz; do
  b=$(basename "$f" .tar.xz)
  ( mkdir -p "stage/$b" && tar -xJf "$f" -C "stage/$b"; echo $? > "$status_dir/$b.rc" ) &
done
wait

fail=0
for f in images/data_*.tar.xz; do
  b=$(basename "$f" .tar.xz)
  rc=$(cat "$status_dir/$b.rc" 2>/dev/null || echo 99)
  echo "shard $b exit=$rc"
  [ "$rc" = "0" ] || fail=1
done
rm -rf "$status_dir"

# Merge into images/ via atomic per-file moves (sequential -> no races even if
# shards share a filename: last writer wins cleanly, never interleaved).
for d in stage/*/; do
  mv -f "$d"patch_*.tif images/ 2>/dev/null
done
rm -rf stage

count=$(ls images/patch_*.tif 2>/dev/null | wc -l)
echo "extracted patch count: $count"
[ -f images/patch_0.tif ] && echo "patch_0.tif OK" || { echo "patch_0.tif MISSING"; fail=1; }
if [ "$count" -lt 90000 ] || [ "$fail" -ne 0 ]; then
  echo "EXTRACT INCOMPLETE (count=$count, fail=$fail)"; exit 1
fi

# Validate every tif by actually reading it; delete any corrupt files so the
# loader skips them. This is the authoritative integrity gate.
source /projects/bgtj/isaaccorley/satclip-env/bin/activate
python /u/isaaccorley/github/satclip/scripts/validate_patches.py --n-jobs 16 --delete
echo "valid patch count: $(ls images/patch_*.tif 2>/dev/null | wc -l)"
echo "EXTRACT DONE"
