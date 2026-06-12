"""Auto-eval monitor: every 25 training epochs, snapshot each cached run's checkpoint
and fire a downstream eval (issue-#6 tasks), appending to eval_results.csv.

Runs as a cheap CPU job. Polls each run's current epoch (from last.ckpt); when a run
crosses a new 25-epoch milestone, copies last.ckpt -> eval_epoch{N}.ckpt and submits a
short GPU eval (eval_once.sh) for it. The pretrained SatCLIP encoder is evaluated once
as a reference. Exits when the training runs finish.
"""
import csv
import json
import os
import shutil
import subprocess
import time

import torch

RUNS = "/projects/bgtj/isaaccorley/satclip_runs"
CACHED = {
    "baseline(rho0)": "vit16l40c_baseline",
    "soft_linear(rho20)": "vit16l40c_linear",
    "soft_exp(rho20)": "vit16l40c_exp",
    "soft_sigmoid(rho20)": "vit16l40c_sigmoid",
}
PRETRAINED = "/u/isaaccorley/github/neuralftw/weights/satclip-location-encoder.pt"
RESULTS = f"{RUNS}/eval_results.csv"
EVAL_SBATCH = "/u/isaaccorley/github/satclip/slurm/eval_once.sh"
STEP, POLL = 25, 180


def epoch_of(ckpt):
    try:
        return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", -1))
    except Exception:
        return -1


def done_set():
    d = set()
    if os.path.exists(RESULTS):
        for r in csv.DictReader(open(RESULTS)):
            d.add((r["label"], r["epoch"]))
    return d


def submit(entries):
    mf = f"{RUNS}/_eval_manifest_{int(time.time())}.json"
    json.dump(entries, open(mf, "w"))
    subprocess.run(["sbatch", EVAL_SBATCH, mf])
    print(f"[{time.strftime('%H:%M')}] submitted eval:", [(e["label"], e["epoch"]) for e in entries], flush=True)


def training_alive():
    out = subprocess.run(["squeue", "-u", "isaaccorley", "-h", "-o", "%j"],
                         capture_output=True, text=True).stdout
    return any(n.startswith("vitc_") for n in out.split())


def main():
    done = done_set()
    if not any(l == "satclip_pretrained(ref)" for l, _ in done):
        submit([{"label": "satclip_pretrained(ref)", "ckpt": PRETRAINED, "epoch": "ref"}])
        done.add(("satclip_pretrained(ref)", "ref"))
    while True:
        pending = []
        for label, d in CACHED.items():
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
        if not training_alive():
            print("training finished; monitor exiting", flush=True)
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
