"""Auto-eval + early-stop monitor for the cached SatCLIP runs.

Every STEP epochs: snapshot each run's checkpoint and fire a downstream eval
(issue-#6 tasks) -> eval_results.csv. Also EARLY-STOP a run once its downstream
performance is clearly past its peak (declined for STOP_PATIENCE milestones) --
so we don't waste epochs once "more training does worse", while runs still
*improving* toward the pretrained reference keep going. Cheap CPU job; exits
when all training runs are done.
"""
import collections
import csv
import json
import os
import shutil
import subprocess
import time

import torch

RUNS = "/projects/bgtj/isaaccorley/satclip_runs"
# label -> (run dir, slurm job name)
CACHED = {
    "baseline_random(rho0)": ("vit16l40c_baseline_random", "vitc_base_rand"),
    "baseline(rho0)": ("vit16l40c_baseline", "vitc_baseline"),
    "soft_linear(rho20)": ("vit16l40c_linear", "vitc_linear"),
    "soft_exp(rho20)": ("vit16l40c_exp", "vitc_exp"),
    "soft_sigmoid(rho20)": ("vit16l40c_sigmoid", "vitc_sigmoid"),
    "soft_linear(rho50)": ("vit16l40c_linear_rho50", "vitc_lin50"),
    "soft_linear(rho100)": ("vit16l40c_linear_rho100", "vitc_lin100"),
    "soft_linear(rho200)": ("vit16l40c_linear_rho200", "vitc_lin200"),
}
PRETRAINED = "/u/isaaccorley/github/neuralftw/weights/satclip-location-encoder.pt"
RESULTS = f"{RUNS}/eval_results.csv"
EVAL_SBATCH = "/u/isaaccorley/github/satclip/slurm/eval_once.sh"
STEP, POLL = 5, 90
STOP_PATIENCE, MIN_EVALS = 3, 5   # cancel after this many declining milestones past peak


def epoch_of(ckpt):
    try:
        return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", -1))
    except Exception:
        return -1


def load_rows():
    return list(csv.DictReader(open(RESULTS))) if os.path.exists(RESULTS) else []


def done_set(rows):
    return {(r["label"], r["epoch"]) for r in rows}


def mean_by_epoch(rows, label):
    by = collections.defaultdict(list)
    for r in rows:
        if r["label"] == label and r["epoch"].isdigit():
            by[int(r["epoch"])].append(float(r["value"]))
    return {e: sum(v) / len(v) for e, v in by.items() if len(v) >= 4}  # need most tasks


def past_peak(rows, label):
    m = mean_by_epoch(rows, label)
    eps = sorted(m)
    if len(eps) < MIN_EVALS:
        return False
    peak = max(eps, key=lambda e: m[e])
    after = [e for e in eps if e > peak]
    return len(after) >= STOP_PATIENCE and all(m[e] < m[peak] for e in after[-STOP_PATIENCE:])


def submit(entries):
    mf = f"{RUNS}/_eval_manifest_{int(time.time())}.json"
    json.dump(entries, open(mf, "w"))
    subprocess.run(["sbatch", EVAL_SBATCH, mf])
    print(f"[{time.strftime('%H:%M')}] eval:", [(e['label'], e['epoch']) for e in entries], flush=True)


def training_alive():
    out = subprocess.run(["squeue", "-u", "isaaccorley", "-h", "-o", "%j"], capture_output=True, text=True).stdout
    return any(n.startswith("vitc_") for n in out.split())


def main():
    rows = load_rows()
    done = done_set(rows)
    stopped = set()
    if not any(l == "satclip_pretrained(ref)" for l, _ in done):
        submit([{"label": "satclip_pretrained(ref)", "ckpt": PRETRAINED, "epoch": "ref"}])
        done.add(("satclip_pretrained(ref)", "ref"))
    while True:
        pending = []
        for label, (d, _) in CACHED.items():
            ck = f"{RUNS}/{d}/checkpoints/last.ckpt"
            if not os.path.exists(ck):
                continue
            ms = (epoch_of(ck) // STEP) * STEP
            if ms >= STEP and (label, str(ms)) not in done:
                snap = f"{RUNS}/{d}/checkpoints/eval_epoch{ms}.ckpt"
                shutil.copyfile(ck, snap)
                pending.append({"label": label, "ckpt": snap, "epoch": str(ms)})
                done.add((label, str(ms)))
        if pending:
            submit(pending)

        # early-stop runs clearly past their downstream peak (keep improvers running)
        rows = load_rows()
        for label, (_, jobname) in CACHED.items():
            if label in stopped:
                continue
            if past_peak(rows, label):
                subprocess.run(["scancel", "--name", jobname, "-u", "isaaccorley"])
                stopped.add(label)
                print(f"[{time.strftime('%H:%M')}] EARLY-STOP {label} (past downstream peak)", flush=True)

        if not training_alive():
            print("training finished; monitor exiting", flush=True)
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
