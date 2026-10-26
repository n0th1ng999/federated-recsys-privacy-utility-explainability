#!/usr/bin/env python3
"""
Runs run_experiment.py for every (dataset, seed) pair in parallel worker processes.

  python experiments/run_sweep.py --workers 6
  python experiments/run_sweep.py --timing --workers 1      # sequential timing runs

--timing writes to results/timing with one process at a time and only the
configurations used in the overhead table, so latencies are not distorted by
concurrent jobs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SEEDS = list(range(42, 52))


def run(job):
    dataset, seed, extra, out = job
    log_dir = os.path.join(REPO_ROOT, out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"{dataset}_seed{seed}.log")
    cmd = [sys.executable, os.path.join(HERE, "run_experiment.py"), "--dataset", dataset, "--seed", str(seed), "--out", out] + extra
    with open(log, "a", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    return dataset, seed, rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", nargs="+", default=["ears", "movielens"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()

    jobs = []
    for seed in args.seeds:
        for ds in args.datasets:
            if args.timing:
                timing_cfgs = ["E1", "E2_s4", "E3", "E4"]
                if ds == "ears":
                    timing_cfgs += ["E3_topk50", "E3_topk20", "E3_topk10", "E3_topk05",
                                    "E1_topk50", "E1_topk20", "E1_topk10", "E1_topk05"]
                jobs.append((ds, seed, ["--skip-xai", "--skip-mia", "--configs", *timing_cfgs], os.path.join("experiments", "results", "timing")))
            else:
                # MovieLens: replication of the main configurations only (no XAI, no Top-K)
                extra = (["--skip-xai", "--configs", "E0", "E1", "E2_s1", "E2_s2", "E2_s4", "E2_s8", "E3", "E4"]
                         if ds == "movielens" else [])
                jobs.append((ds, seed, extra, os.path.join("experiments", "results", "main")))

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ds, seed, rc in ex.map(run, jobs):
            print(f"{ds} seed {seed}: {'ok' if rc == 0 else f'FAILED rc={rc}'}", flush=True)
            if rc != 0:
                failed.append((ds, seed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
