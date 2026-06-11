"""Benchmark S2Geo DataLoader throughput across storage formats x worker counts.

Re-encodes a subset of patches into LZW (original), uncompressed, and ZSTD
TIFFs on node-local /tmp (isolating decode cost from NFS), then times the real
S2Geo dataset + training transform to find samples/sec. Tells us whether to
re-encode the dataset and how many workers to use before the multi-day runs.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch

sys.path.insert(0, "/u/isaaccorley/github/satclip/satclip")
from datamodules.s2geo_dataset import S2Geo
from datamodules.transforms import get_pretrained_s2_train_transform

SRC = "/projects/bgtj/isaaccorley/s2-100k-tg"
WORK = os.environ.get("BENCH_DIR", "/tmp/bench")
N = int(os.environ.get("BENCH_N", "4000"))
BATCH = 512
WORKER_GRID = [8, 16, 32, 64]


def zstd_supported():
    try:
        prof = dict(driver="GTiff", height=4, width=4, count=1, dtype="uint16", compress="zstd")
        p = os.path.join(WORK, "_zstd_probe.tif")
        with rasterio.open(p, "w", **prof) as d:
            d.write(np.zeros((1, 4, 4), "uint16"))
        os.remove(p)
        return True
    except Exception as e:
        print("zstd not supported:", e, flush=True)
        return False


def reencode(fmts, idx):
    df = pd.read_csv(os.path.join(SRC, "index.csv"))
    # keep first N rows whose source patch exists
    rows = []
    for _, r in df.iterrows():
        if os.path.exists(os.path.join(SRC, "images", r["fn"])):
            rows.append(r)
        if len(rows) >= N:
            break
    sub = pd.DataFrame(rows)
    for fmt, opts in fmts.items():
        d = os.path.join(WORK, fmt)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        sub.to_csv(os.path.join(d, "index.csv"), index=False)
        for fn in sub["fn"]:
            with rasterio.open(os.path.join(SRC, "images", fn)) as r:
                data = r.read()
                prof = r.profile
            prof.update(**opts)
            with rasterio.open(os.path.join(d, "images", fn), "w", **prof) as w:
                w.write(data)
        sz = sum(os.path.getsize(os.path.join(d, "images", fn)) for fn in sub["fn"])
        print(f"  re-encoded {fmt}: {len(sub)} patches, {sz/1e6:.0f} MB total ({sz/len(sub)/1e3:.0f} KB/patch)", flush=True)


def bench(fmt, num_workers, n_batches=10):
    d = os.path.join(WORK, fmt)
    ds = S2Geo(root=d, transform=get_pretrained_s2_train_transform(256), mode="both")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=BATCH, num_workers=num_workers, shuffle=True,
        persistent_workers=True, drop_last=True, prefetch_factor=4,
    )

    # cycle batches (subset has only ~7 full batches); persistent workers mean
    # re-iterating doesn't respawn, so the timed loop measures steady state.
    def gen():
        while True:
            for b in dl:
                yield b

    g = gen()
    for _ in range(3):  # warmup: spawn workers + fill prefetch
        next(g)
    t0 = time.time()
    seen = 0
    for _ in range(n_batches):
        b = next(g)
        seen += b["image"].shape[0]
    dt = time.time() - t0
    del g, dl
    return seen / dt


def main():
    os.makedirs(WORK, exist_ok=True)
    fmts = {"lzw": dict(compress="lzw"), "none": dict(compress="none")}
    if zstd_supported():
        fmts["zstd"] = dict(compress="zstd", zstd_level=9)
    print(f"formats: {list(fmts)}; N={N}; re-encoding to {WORK} ...", flush=True)
    reencode(fmts, N)

    print(f"\n{'format':>8} | " + " | ".join(f"nw={w:<3}" for w in WORKER_GRID) + "   (samples/sec)", flush=True)
    for fmt in fmts:
        cells = []
        for w in WORKER_GRID:
            try:
                cells.append(f"{bench(fmt, w):7.0f}")
            except Exception as e:
                import traceback; traceback.print_exc()
                cells.append(f"ERR:{type(e).__name__}")
        print(f"{fmt:>8} | " + " | ".join(cells), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
