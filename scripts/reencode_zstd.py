"""Re-encode the S2-100K patches from LZW to ZSTD TIFF (lossless).

Benchmarked at ~half the disk of LZW and +25% DataLoader throughput. Reading
each source patch also validates it; the output is read-tested separately by
validate_patches.py. Parallel across CPU cores with joblib; resumable (skips
patches already written).
"""
import os
import shutil
import sys

import rasterio
from joblib import Parallel, delayed

SRC = os.environ.get("SRC", "/projects/bgtj/isaaccorley/s2-100k-tg")
DST = os.environ.get("DST", "/projects/bgtj/isaaccorley/s2-100k-tg-zstd")
N_JOBS = int(os.environ.get("N_JOBS", "64"))


def reencode(fn):
    dst = os.path.join(DST, "images", fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return  # resumable
    with rasterio.open(os.path.join(SRC, "images", fn)) as r:
        data = r.read()
        prof = r.profile
    prof.update(compress="zstd", zstd_level=9)
    tmp = dst + ".tmp"
    with rasterio.open(tmp, "w", **prof) as w:
        w.write(data)
    os.replace(tmp, dst)  # atomic -> no half-written files on failure


def main():
    os.makedirs(os.path.join(DST, "images"), exist_ok=True)
    # index.csv may be a symlink into the HF cache; copy the real contents.
    shutil.copyfile(os.path.realpath(os.path.join(SRC, "index.csv")),
                    os.path.join(DST, "index.csv"))

    files = [f for f in os.listdir(os.path.join(SRC, "images")) if f.endswith(".tif")]
    print(f"re-encoding {len(files)} patches: {SRC} -> {DST} (zstd, {N_JOBS} jobs)", flush=True)
    Parallel(n_jobs=N_JOBS, batch_size=64)(delayed(reencode)(f) for f in files)

    out = len([f for f in os.listdir(os.path.join(DST, "images")) if f.endswith(".tif")])
    print(f"wrote {out} patches", flush=True)
    if out != len(files):
        print(f"COUNT MISMATCH: {out} != {len(files)}", flush=True)
        sys.exit(1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
