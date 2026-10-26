#!/usr/bin/env python3
"""
Single entry point for the experiments.

One invocation = one dataset x one seed. For every configuration it trains the
model, evaluates ranking on the temporal holdout, runs both membership-inference
attacks, optionally runs the explanation protocol, and writes one JSON per
(configuration, seed) to <out>/<dataset>/<config>/seed_<seed>.json together with
provenance (code hash, library versions, split description, full config).

Example:
  python experiments/run_experiment.py --dataset ears --seed 42
  python experiments/run_experiment.py --dataset movielens --seed 42 --skip-xai
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss  # noqa: F401  (must be imported before torch on macOS)
import argparse
import dataclasses
import hashlib
import json
import platform
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.attacks.membership import run_membership_attacks
from experiments.centralized.trainer import run_centralized_experiment
from experiments.config import get_all_experiment_configs, get_quick_experiment_configs
from experiments.data.movielens_adapter import ensure_movielens_data
from experiments.data.partitioner import prepare_experiment_data
from experiments.evaluation.ranking_metrics import evaluate_model_ranking, evaluate_score_baseline
from experiments.evaluation.xai_protocol import agreement_with_reference, run_xai_protocol
from experiments.federated.fl_runner import run_federated_experiment
from experiments.federated.he_engine import TENSEAL_VERSION

DATA_DIRS = {"ears": "data", "movielens": os.path.join("data", "movielens")}
CODE_DIRS = ["experiments", "recommender_core", "support"]
CODE_FILES = ["api_service/preprocess.py"]


def code_fingerprint() -> str:
    h = hashlib.sha256()
    paths = []
    for d in CODE_DIRS:
        for root, _, files in os.walk(os.path.join(REPO_ROOT, d)):
            if "results" in root.split(os.sep) or "__pycache__" in root or "legacy" in root.split(os.sep):
                continue
            paths += [os.path.join(root, f) for f in files if f.endswith(".py")]
    paths += [os.path.join(REPO_ROOT, f) for f in CODE_FILES]
    for p in sorted(paths):
        h.update(os.path.relpath(p, REPO_ROOT).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def environment_info() -> Dict[str, Any]:
    import dp_accounting, lime, shap, sklearn
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "tenseal": TENSEAL_VERSION,
        "dp_accounting": getattr(dp_accounting, "__version__", "0.6.0"),
        "shap": shap.__version__,
        "lime": getattr(lime, "__version__", "0.2.0.1"),
        "sklearn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATA_DIRS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument("--skip-xai", action="store_true")
    ap.add_argument("--skip-mia", action="store_true", help="timing runs: skip membership inference")
    ap.add_argument("--xai-instances", type=int, default=30)
    ap.add_argument("--out", default=os.path.join("experiments", "results", "main"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    device = torch.device("cpu")
    seed = args.seed
    if args.dataset == "movielens":
        ensure_movielens_data(DATA_DIRS["movielens"])

    suite = get_quick_experiment_configs(seed) if args.quick else get_all_experiment_configs(seed)
    if args.configs:
        unknown = sorted(set(args.configs) - set(suite))
        if unknown:
            ap.error(f"unknown configuration(s) {unknown}; valid: {sorted(suite)}")
        suite = {k: v for k, v in suite.items() if k in args.configs}
    # E0 first: its explanations are the reference for the agreement measure.
    keys = sorted(suite, key=lambda k: (k != "E0", k))

    out_root = os.path.join(args.out, args.dataset)
    pending = [k for k in keys if args.overwrite or not os.path.exists(os.path.join(out_root, k, f"seed_{seed}.json"))]
    if not pending:
        print("All results present; nothing to do.")
        return 0
    if not args.skip_xai and "E0" not in pending and "E0" in keys:
        pending = ["E0"] + pending  # recompute reference explanations

    first = suite[keys[0]]
    t0 = time.perf_counter()
    bundle = prepare_experiment_data(
        data_dir=DATA_DIRS[args.dataset],
        num_clients=first.fl.num_clients,
        eval_users_count=first.eval.eval_users,
        candidates_per_user=first.eval.candidates_per_user,
        seed=seed,
        mia_nonmember_users=20 if not args.quick else 4,
    )
    split = bundle.eval_data["split"]
    print(f"[data] {args.dataset} seed={seed} cutoff={split['cutoff_date']} "
          f"train_clients={len(bundle.client_partitions)} eval_users={len(bundle.eval_data['eval_user_ids'])} "
          f"({time.perf_counter() - t0:.1f}s)")

    # XAI inputs (fixed per seed, identical for every configuration)
    rng = np.random.default_rng(seed)
    ds = bundle.full_dataset
    train_pool = [i for idxs in bundle.client_partitions.values() for i in idxs]
    eval_pool = [i for idxs in bundle.eval_data["eval_user_samples"].values() for i in idxs]
    holdout_pool = [i for idxs in bundle.eval_data["mia_record_holdout_samples"].values() for i in idxs]
    n_inst = min(args.xai_instances, len(eval_pool))
    xai_instances = [ds[int(i)] for i in rng.choice(eval_pool, size=n_inst, replace=False)]
    xai_background = [ds[int(i)] for i in rng.choice(train_pool, size=min(20, len(train_pool)), replace=False)]
    xai_reference = [ds[int(i)] for i in rng.choice(train_pool, size=min(200, len(train_pool)), replace=False)]
    xai_heldout = [ds[int(i)] for i in rng.choice(holdout_pool, size=min(100, len(holdout_pool)), replace=False)]

    # Non-learned references on the same candidates (written once per seed)
    baselines_path = os.path.join(_ensure(out_root, "baselines"), f"seed_{seed}.json")
    if args.overwrite or not os.path.exists(baselines_path):
        import pandas as pd
        d = DATA_DIRS[args.dataset]
        orders = pd.read_csv(os.path.join(d, "orders.csv"), parse_dates=["OrderDate"])
        items = pd.read_csv(os.path.join(d, "order_items.csv")).merge(orders[["OrderID", "OrderDate"]], on="OrderID")
        popularity = items[items.OrderDate <= pd.Timestamp(split["cutoff_date"])].ProductID.value_counts().to_dict()
        kw = dict(eval_user_samples=bundle.eval_data["eval_user_samples"], dataset=ds,
                  relevance_threshold=bundle.eval_data["relevance_threshold"], seed=seed)
        with open(baselines_path, "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "split": split,
                       "random": evaluate_score_baseline(item_scores={}, random_repeats=100, **kw),
                       "popularity_pre_cutoff": evaluate_score_baseline(item_scores=popularity, **kw)}, f, indent=1)

    provenance = {
        "code_sha256": code_fingerprint(),
        "environment": environment_info(),
        "dataset": args.dataset,
        "seed": seed,
        "split": split,
    }
    reference_attr = None

    for key in pending:
        cfg = suite[key]
        print(f"\n=== [{args.dataset} | seed {seed}] {key}: {cfg.description}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        t_train = time.perf_counter()
        if cfg.is_federated:
            model, tracker = run_federated_experiment(cfg, bundle, device)
            participation = tracker.client_participation
        else:
            model, tracker = run_centralized_experiment(cfg, bundle, device)
            participation = {c: 1 for c in bundle.client_partitions}
        train_wall = time.perf_counter() - t_train

        t_eval = time.perf_counter()
        ranking = evaluate_model_ranking(
            model=model,
            eval_user_samples=bundle.eval_data["eval_user_samples"],
            dataset=ds,
            device=device,
            k_values=cfg.eval.k_values,
            primary_k=cfg.eval.primary_k,
            relevance_threshold=bundle.eval_data["relevance_threshold"],
        )
        tracker.inference_time_sec = time.perf_counter() - t_eval
        print(f"  nDCG@10={ranking.get('ndcg@10', 0):.4f}  Recall@10={ranking.get('recall@10', 0):.4f}")

        mia = None
        if not args.skip_mia:
            mia = run_membership_attacks(model, bundle, participation, device, seed)
            print(f"  MIA client AUC={mia['client_participation']['auc']:.3f}  record AUC={mia['record_membership']['auc']:.3f}")

        xai = None
        if not args.skip_xai and cfg.fl.topk_mode == "none":
            t_x = time.perf_counter()
            xai = run_xai_protocol(model, device, xai_instances, xai_background, xai_reference, xai_heldout, seed)
            attr = xai.pop("_shap_attributions")
            if key == "E0":
                reference_attr = attr
                np.save(os.path.join(_ensure(out_root, "E0"), f"shap_attributions_seed_{seed}.npy"), attr)
            elif reference_attr is not None:
                xai["agreement_with_E0_shap"] = agreement_with_reference(attr, reference_attr, seed)
            np.save(os.path.join(_ensure(out_root, key), f"shap_attributions_seed_{seed}.npy"), attr)
            print(f"  XAI protocol {time.perf_counter() - t_x:.1f}s")

        result = {
            "configuration": key,
            "config": dataclasses.asdict(cfg),
            "provenance": provenance,
            "train_wall_clock_sec": train_wall,
            "recommendation_metrics": ranking,
            "performance_metrics": tracker.to_dict(),
            "membership_inference": mia,
            "explainability": xai,
        }
        path = os.path.join(_ensure(out_root, key), f"seed_{seed}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jsonable(result), f, indent=1)
        print(f"  -> {path}")
    return 0


def _ensure(root: str, key: str) -> str:
    d = os.path.join(root, key)
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    raise SystemExit(main())
