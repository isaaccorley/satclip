"""Evaluate SatCLIP location-encoder checkpoints on the official downstream tasks.

Reuses neuralftw's harness (issue-#6 datasets, embed_satclip, ridge/logistic probe).
embed_satclip loads our Lightning ckpts directly. Reports R2 (regression) / accuracy
(classification) per task. Driven by a JSON manifest of {label, ckpt, epoch} and
APPENDS rows to a results CSV, so the auto-eval monitor can call it per milestone.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, "/u/isaaccorley/github/neuralftw/src")
import torch
from neuralftw.eval import benchmarks as B
from neuralftw.eval.embedders import embed_satclip
from neuralftw.eval.probe import evaluate_tasks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON list of {label, ckpt, epoch}")
    ap.add_argument("--out", default="/projects/bgtj/isaaccorley/satclip_runs/eval_results.csv")
    args = ap.parse_args()
    entries = json.load(open(args.manifest))
    benches = B.load_satclip_official()

    new_rows = []
    for bench in benches:
        metric = "R2" if bench.task_type == "regression" else "acc"
        for e in entries:
            ck = e["ckpt"]
            if not os.path.exists(ck):
                print(f"  [skip] {e['label']} ckpt missing: {ck}", flush=True)
                continue
            feats = embed_satclip(ck, bench.lat, bench.lon, device=DEVICE)
            res = evaluate_tasks(feats, bench.tasks, task_type=bench.task_type, device=DEVICE,
                                 clf_probe="ridge", lat=bench.lat, lon=bench.lon)
            val = res if isinstance(res, (int, float)) else list(res.values())[0]
            new_rows.append({"label": e["label"], "epoch": e.get("epoch", ""), "task": bench.name,
                             "metric": metric, "value": round(float(val), 4)})
            print(f"  {bench.name:22s} {e['label']:22s} ep{e.get('epoch','')} {metric}={float(val):.4f}", flush=True)

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "epoch", "task", "metric", "value"])
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    print(f"appended {len(new_rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
