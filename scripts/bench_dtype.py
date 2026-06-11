"""Test the IPC-ceiling hypothesis: does shrinking the worker->main payload
(uint16 12-band raw vs float32 13-band transformed) raise DataLoader throughput?

If the lean uint16 loader is much faster, the bottleneck is the per-batch
collate/IPC memcpy, and moving the transform to the GPU is the real speedup.
Reads a subset staged on node-local /tmp to isolate from NFS.
"""
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch

sys.path.insert(0, "/u/isaaccorley/github/satclip/satclip")
from datamodules.s2geo_dataset import S2Geo
from datamodules.transforms import get_pretrained_s2_train_transform

SRC = "/projects/bgtj/isaaccorley/s2-100k-tg-zstd"
WORK = os.environ.get("BENCH_DIR", "/tmp/bench_dtype")
N = int(os.environ.get("BENCH_N", "2500"))
BATCH = 512


def stage():
    os.makedirs(os.path.join(WORK, "images"), exist_ok=True)
    df = pd.read_csv(os.path.join(SRC, "index.csv"))
    rows = []
    for _, r in df.iterrows():
        s = os.path.join(SRC, "images", r["fn"])
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(WORK, "images", r["fn"]))
            rows.append(r)
        if len(rows) >= N:
            break
    pd.DataFrame(rows).to_csv(os.path.join(WORK, "index.csv"), index=False)
    print(f"staged {len(rows)} zstd patches to {WORK}", flush=True)


# Lean transform: return raw uint16 12-band (what a GPU-side pipeline would get).
def lean_transform(sample):
    img = sample["image"]  # uint16 numpy [12,256,256]
    return dict(image=torch.from_numpy(np.ascontiguousarray(img)), point=sample["point"])


def bench(transform, label, num_workers=8, pin=False, n_batches=10):
    ds = S2Geo(root=WORK, transform=transform, mode="both")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=BATCH, num_workers=num_workers, shuffle=True,
        persistent_workers=True, drop_last=True, prefetch_factor=4, pin_memory=pin,
    )

    def gen():
        while True:
            for b in dl:
                yield b

    g = gen()
    for _ in range(3):
        next(g)
    t0 = time.time()
    seen = 0
    for _ in range(n_batches):
        b = next(g)
        seen += b["image"].shape[0]
    dt = time.time() - t0
    mb = b["image"].element_size() * b["image"].nelement() / 1e6
    del g, dl
    print(f"  {label:28} {seen/dt:7.0f} samples/s   (batch {tuple(b['image'].shape)} {b['image'].dtype}, {mb:.0f} MB/batch)", flush=True)


def main():
    stage()
    print("\nthroughput (nw=8, /tmp):", flush=True)
    bench(get_pretrained_s2_train_transform(256), "current (float32 13-band)", pin=False)
    bench(lean_transform, "lean uint16 12-band raw", pin=False)
    bench(lean_transform, "lean uint16 + pin_memory", pin=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
