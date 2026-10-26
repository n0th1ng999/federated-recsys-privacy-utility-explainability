#!/usr/bin/env python3
"""
How far do the removal-curve interventions move inputs away from the data?

Model-free and post hoc: for each seed it rebuilds the explained instances, the
reference pool and the held-out rows exactly as run_experiment.py drew them, takes the
saved SHAP attributions of each configuration to recover the removal order, and
replays the marginal and conditional replacements at k = 3 and k = 10.

Distance: Euclidean distance to the nearest reference row after standardising each
feature by max(SD over the reference pool, 0.01). The floor matters: a feature that is
constant in the reference pool would otherwise be divided by ~0 (the first version of
this summary used 1e-8 and was dominated by one such feature in one seed). Features
lie in [0, 1], so 0.01 is a small fraction of their range.

Reported per configuration: median distance of genuine held-out rows, of the
unmodified instances, and of intervened inputs, and the median increase over the
unmodified instance.

Writes <results>/<dataset>/plausibility/seed_<n>.json
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss  # noqa: F401
import argparse
import json
import sys

import numpy as np

REPO_ROOT = os.environ.get("RECSYS_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, REPO_ROOT)

from experiments.data.partitioner import prepare_experiment_data
from experiments.evaluation.xai_evaluator import extract_tabular_features_from_sample as features
from experiments.evaluation.xai_protocol import Replacer
from experiments.run_experiment import DATA_DIRS

SD_FLOOR = 0.01
CONFIGS = ["E0", "E1", "E2_s1", "E2_s2", "E2_s4", "E2_s8", "E3", "E4"]


def nn_dist(rows, ref, mu, sd):
    z = (rows - mu) / sd
    rz = (ref - mu) / sd
    return np.sqrt(((z[:, None, :] - rz[None, :, :]) ** 2).sum(-1)).min(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ears")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--results", default=os.path.join(REPO_ROOT, "experiments", "results", "main"))
    ap.add_argument("--draws", type=int, default=10)
    args = ap.parse_args()
    for seed in args.seeds:
        out = os.path.join(args.results, args.dataset, "plausibility", f"seed_{seed}.json")
        if os.path.exists(out):
            continue
        bundle = prepare_experiment_data(data_dir=os.path.join(REPO_ROOT, DATA_DIRS[args.dataset]), num_clients=20,
                                         eval_users_count=50, candidates_per_user=50, seed=seed)
        # identical draw sequence to run_experiment.py
        rng = np.random.default_rng(seed)
        ds = bundle.full_dataset
        train_pool = [i for idxs in bundle.client_partitions.values() for i in idxs]
        eval_pool = [i for idxs in bundle.eval_data["eval_user_samples"].values() for i in idxs]
        holdout_pool = [i for idxs in bundle.eval_data["mia_record_holdout_samples"].values() for i in idxs]
        inst = np.stack([features(ds[int(i)]) for i in rng.choice(eval_pool, size=min(30, len(eval_pool)), replace=False)])
        rng.choice(train_pool, size=min(20, len(train_pool)), replace=False)  # background (unused here)
        ref = np.stack([features(ds[int(i)]) for i in rng.choice(train_pool, size=min(200, len(train_pool)), replace=False)])
        held = np.stack([features(ds[int(i)]) for i in rng.choice(holdout_pool, size=min(100, len(holdout_pool)), replace=False)])

        mu, sd = ref.mean(0), np.maximum(ref.std(0), SD_FLOOR)
        replacer = Replacer(ref)  # same replacement mechanics as the protocol
        d_inst = nn_dist(inst, ref, mu, sd)
        res = {"seed": seed, "sd_floor": SD_FLOOR,
               "heldout_median": float(np.median(nn_dist(held, ref, mu, sd))),
               "instance_median": float(np.median(d_inst)), "configs": {}}
        prng = np.random.default_rng(seed + 99)
        for cfg in CONFIGS:
            path = os.path.join(args.results, args.dataset, cfg, f"shap_attributions_seed_{seed}.npy")
            if not os.path.exists(path):
                continue
            attr = np.load(path)
            out_cfg = {}
            for scheme in ("marginal", "conditional"):
                for k in (3, 10):
                    dist, inc = [], []
                    for i, x in enumerate(inst):
                        order = np.argsort(-np.abs(attr[i]))
                        rows = np.stack([replacer.apply(x, order[:k], scheme, prng) for _ in range(args.draws)])
                        dd = nn_dist(rows, ref, mu, sd)
                        dist.extend(dd.tolist())
                        inc.extend((dd - d_inst[i]).tolist())
                    out_cfg[f"{scheme}_k{k}"] = {"median": float(np.median(dist)), "median_increase": float(np.median(inc))}
            res["configs"][cfg] = out_cfg
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        e1 = res["configs"].get("E1", {})
        print(f"seed {seed}: heldout {res['heldout_median']:.2f} instance {res['instance_median']:.2f} | E1 "
              + " ".join(f"{k} {v['median']:.2f} (+{v['median_increase']:.2f})" for k, v in e1.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
