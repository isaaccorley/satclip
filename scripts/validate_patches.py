"""Validate every patch tif by actually opening + reading it with rasterio.

Files that fail (truncated, corrupt header, wrong format) are reported and,
with --delete, removed so the data loader's existence check skips them.
Parallel across CPU cores with joblib.
"""
import argparse
import glob
import os

import rasterio
from joblib import Parallel, delayed


def check(path):
    try:
        with rasterio.open(path) as f:
            f.read(1)  # force a real read, not just header parse
        return None
    except (rasterio.errors.RasterioIOError, rasterio.errors.RasterioError) as e:
        return (path, str(e).splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/projects/bgtj/isaaccorley/s2-100k/images")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "patch_*.tif")))
    print(f"validating {len(files)} files with rasterio...", flush=True)

    results = Parallel(n_jobs=args.n_jobs, batch_size=256)(
        delayed(check)(p) for p in files
    )
    bad = [r for r in results if r is not None]

    print(f"valid: {len(files) - len(bad)}   corrupt: {len(bad)}", flush=True)
    for path, msg in bad[:50]:
        print(f"  CORRUPT {os.path.basename(path)}: {msg}", flush=True)
    if len(bad) > 50:
        print(f"  ... and {len(bad) - 50} more", flush=True)

    if args.delete:
        for path, _ in bad:
            os.remove(path)
        print(f"deleted {len(bad)} corrupt files", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
