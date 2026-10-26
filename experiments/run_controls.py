#!/usr/bin/env python3
"""
Controls for interpreting the DP results.

For every dataset and seed, on the same split and candidates as run_experiment.py:
  * untrained: the global model at initialisation (same seed as every federated run);
  * noise_only_s<sigma>: the initial model plus exactly the noise E2 adds over T rounds
    (T independent N(0, (sigma*C/m)^2) draws per coordinate), with NO client updates.

If a DP configuration is indistinguishable from its noise-only control, the clients'
clipped updates contributed no detectable signal to ranking quality: per round the
noise on the mean has norm about sigma*C/m*sqrt(d), against at most C for the signal.

Writes <out>/<dataset>/controls/seed_<n>.json
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss  # noqa: F401
import argparse
import json
import math
import sys

import numpy as np
import torch

REPO_ROOT = os.environ.get("RECSYS_REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, REPO_ROOT)

from experiments.config import E2_NOISE_MULTIPLIERS, FLConfig
from experiments.data.movielens_adapter import ensure_movielens_data
from experiments.data.partitioner import prepare_experiment_data
from experiments.evaluation.ranking_metrics import evaluate_model_ranking
from experiments.run_experiment import DATA_DIRS, code_fingerprint
from recommender_core.Core_Model import CoreHybridRecSys


def evaluate(model, bundle):
    r = evaluate_model_ranking(model, bundle.eval_data["eval_user_samples"], bundle.full_dataset, torch.device("cpu"),
                               relevance_threshold=bundle.eval_data["relevance_threshold"])
    return {k: v for k, v in r.items() if k != "per_user"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ears", "movielens"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--noise-draws", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "experiments", "results", "main"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    fl = FLConfig()
    C = 1.0
    for ds in args.datasets:
        if ds == "movielens":
            ensure_movielens_data(os.path.join(REPO_ROOT, DATA_DIRS["movielens"]))
        for seed in args.seeds:
            path = os.path.join(args.out, ds, "controls", f"seed_{seed}.json")
            if os.path.exists(path):
                continue
            bundle = prepare_experiment_data(data_dir=os.path.join(REPO_ROOT, DATA_DIRS[ds]), num_clients=fl.num_clients,
                                             eval_users_count=50, candidates_per_user=50, seed=seed)
            torch.manual_seed(seed)  # identical to the initialisation used by run_experiment / fl_runner
            init = CoreHybridRecSys(branch_dims=bundle.branch_dims)
            res = {"seed": seed, "dataset": ds, "untrained": evaluate(init, bundle), "noise_only": {},
                   "num_parameters": int(sum(p.numel() for p in init.state_dict().values())),
                   "code_sha256": code_fingerprint()}
            d = res["num_parameters"]
            for sg in E2_NOISE_MULTIPLIERS:
                std_total = math.sqrt(fl.num_rounds) * sg * C / fl.clients_per_round
                draws = []
                for k in range(args.noise_draws):
                    g = torch.Generator().manual_seed(seed * 1000 + k)
                    state = {key: (v + torch.randn(v.shape, generator=g) * std_total).to(v.dtype) if v.is_floating_point() else v
                             for key, v in init.state_dict().items()}
                    m = CoreHybridRecSys(branch_dims=bundle.branch_dims)
                    m.load_state_dict(state)
                    draws.append(evaluate(m, bundle))
                res["noise_only"][f"s{sg:g}"] = {
                    "per_draw_ndcg@10": [x["ndcg@10"] for x in draws],
                    "ndcg@10": float(np.mean([x["ndcg@10"] for x in draws])),
                    "recall@10": float(np.mean([x["recall@10"] for x in draws])),
                    "noise_norm_per_round": sg * C / fl.clients_per_round * math.sqrt(d),
                }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            print(f"{ds} seed {seed}: untrained {res['untrained']['ndcg@10']:.3f} | "
                  + " ".join(f"{k} {v['ndcg@10']:.3f}" for k, v in res["noise_only"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
