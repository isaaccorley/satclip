"""Full-dataset pairwise spatial analysis for S2-100K.

Computes, over ALL 100k points (not a sample):
  - exact nearest-neighbor distance distribution
  - exact count of point-pairs within each radius r (km)
  - expected number of such pairs landing in a random training batch

Uses a haversine BallTree; the radius queries are chunked across CPU cores
with joblib. Earth radius matches model.pairwise_haversine_dist (6378 km).
"""
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6378.0
RADII_KM = [1, 10, 50, 100, 300, 500, 1000, 2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/projects/bgtj/isaaccorley/s2-100k/index.csv")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    n = len(df)
    print(f"points: {n}", flush=True)

    # BallTree haversine wants [lat, lon] in radians.
    latlon = np.deg2rad(df[["lat", "lon"]].to_numpy())
    tree = BallTree(latlon, metric="haversine")

    # --- exact nearest-neighbor distance for every point ---
    nn_dist_rad, _ = tree.query(latlon, k=2)  # col 0 = self (0)
    nn_km = nn_dist_rad[:, 1] * EARTH_RADIUS_KM
    pct = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 100]
    print("\n=== nearest-neighbor distance (km), full dataset ===", flush=True)
    for p in pct:
        print(f"  p{p:<5} {np.percentile(nn_km, p):10.3f}", flush=True)
    print(f"  mean  {nn_km.mean():10.3f}", flush=True)

    # --- exact pair counts within each radius (chunked, parallel) ---
    radii_rad = [r / EARTH_RADIUS_KM for r in RADII_KM]
    starts = list(range(0, n, args.chunk))

    def count_chunk(s):
        q = latlon[s : s + args.chunk]
        # count_only returns, per query point, #neighbors within r (incl self)
        return np.array([tree.query_radius(q, rr, count_only=True).sum() for rr in radii_rad])

    totals = Parallel(n_jobs=args.n_jobs)(delayed(count_chunk)(s) for s in starts)
    incl_self = np.sum(totals, axis=0)  # ordered counts incl self, double-counts pairs

    total_pairs = n * (n - 1) // 2
    # probability a fixed pair both land in one random batch of size B:
    B = args.batch_size
    p_in_batch = (B * (B - 1)) / (n * (n - 1))

    print(f"\n=== pairs within radius (full {n} points, {total_pairs:,} total pairs) ===", flush=True)
    print(f"{'r(km)':>7} {'#pairs':>14} {'%all pairs':>12} {'exp/batch(512)':>16}", flush=True)
    prev = 0
    for r, c in zip(RADII_KM, incl_self):
        pairs = (int(c) - n) // 2  # remove self-matches, undouble
        print(
            f"{r:>7} {pairs:>14,} {100*pairs/total_pairs:>11.5f}% {pairs*p_in_batch:>16.4f}",
            flush=True,
        )
        prev = pairs
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
