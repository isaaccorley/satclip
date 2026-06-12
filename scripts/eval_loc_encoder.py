"""Evaluate trained SatCLIP location encoders on the official downstream tasks.

Reuses neuralftw's eval harness (the SatCLIP issue-#6 datasets, embed_satclip,
ridge/logistic probe) — no rebuild. embed_satclip loads our Lightning ckpts
directly. Reports R2 (regression) / accuracy (classification) per task, with the
pretrained SatCLIP encoder as a reference column.
"""
import json
import os
import sys

sys.path.insert(0, "/u/isaaccorley/github/neuralftw/src")
import numpy as np
import torch
from neuralftw.eval import benchmarks as B
from neuralftw.eval.embedders import embed_satclip
from neuralftw.eval.probe import evaluate_tasks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = "/projects/bgtj/isaaccorley/satclip_runs"
PRETRAINED = "/u/isaaccorley/github/neuralftw/weights/satclip-location-encoder.pt"

CKPTS = {
    "satclip_pretrained(ref)": PRETRAINED,
    "baseline(rho0)": f"{RUNS}/baseline_proxy/checkpoints/last.ckpt",
    "soft_linear(rho20)": f"{RUNS}/softloss_linear/checkpoints/last.ckpt",
    "soft_sigmoid(rho20)": f"{RUNS}/softloss_sigmoid/checkpoints/last.ckpt",
}


def main():
    print(f"device={DEVICE}", flush=True)
    benches = B.load_satclip_official()
    results = {}
    for bench in benches:
        metric = "R2" if bench.task_type == "regression" else "acc"
        print(f"\n=== {bench.name} ({bench.task_type}, n={len(bench.lat)}) [{metric}] ===", flush=True)
        results[bench.name] = {}
        for name, ck in CKPTS.items():
            if not os.path.exists(ck):
                print(f"  {name:24s} ckpt missing -> skip", flush=True)
                continue
            feats = embed_satclip(ck, bench.lat, bench.lon, device=DEVICE)
            res = evaluate_tasks(
                feats, bench.tasks, task_type=bench.task_type, device=DEVICE,
                clf_probe="ridge", lat=bench.lat, lon=bench.lon,
            )
            val = res if isinstance(res, (int, float)) else list(res.values())[0]
            val = val if isinstance(val, (int, float)) else float(np.mean(list(val.values()))) if isinstance(val, dict) else val
            results[bench.name][name] = val
            print(f"  {name:24s} {metric}={val:.4f}", flush=True)

    print("\n================ SUMMARY ================", flush=True)
    print(json.dumps(results, indent=2, default=float), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
