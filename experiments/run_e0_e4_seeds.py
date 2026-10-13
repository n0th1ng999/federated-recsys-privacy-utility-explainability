#!/usr/bin/env python3
"""
Runs E0 (Centralized baseline) and E4 (Hybrid DP+HE) across seeds 45 to 51,
combining them with historical seeds 42..44 and existing E1, E2, E3 results
to produce a complete, unified 10-seed benchmark for all 7 configurations.
"""
from __future__ import annotations

import os
import sys
import json
import time
from typing import Any, Dict
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.config import get_e0_centralized_config, get_e4_fl_dp_he_config
from experiments.data.partitioner import prepare_experiment_data
from experiments.centralized.trainer import run_centralized_experiment
from experiments.federated.fl_runner import run_federated_experiment
from experiments.evaluation.ranking_metrics import evaluate_model_ranking

RESULTS_FILE = "experiments/results/10seeds_benchmark/ten_seeds_results.json"
SUMMARY_FILE = "experiments/results/10seeds_benchmark/ten_seeds_statistical_summary.json"

HISTORICAL_E0_E4 = {
    42: {
        "E0": {"ndcg@10": 0.5280, "recall@10": 0.4033, "precision@10": 0.8380, "map@10": 0.9412, "hit_rate@10": 0.9000, "train_time_sec": 27.32},
        "E4": {"ndcg@10": 0.4283, "recall@10": 0.2331, "precision@10": 0.4940, "map@10": 0.4279, "hit_rate@10": 0.8600, "train_time_sec": 51.15},
    },
    43: {
        "E0": {"ndcg@10": 0.5235, "recall@10": 0.3698, "precision@10": 0.8120, "map@10": 0.9090, "hit_rate@10": 0.9000, "train_time_sec": 26.85},
        "E4": {"ndcg@10": 0.4741, "recall@10": 0.2706, "precision@10": 0.5220, "map@10": 0.5237, "hit_rate@10": 0.8800, "train_time_sec": 50.80},
    },
    44: {
        "E0": {"ndcg@10": 0.5268, "recall@10": 0.3802, "precision@10": 0.8160, "map@10": 0.9143, "hit_rate@10": 0.9000, "train_time_sec": 27.80},
        "E4": {"ndcg@10": 0.3988, "recall@10": 0.1392, "precision@10": 0.3940, "map@10": 0.4261, "hit_rate@10": 0.5600, "train_time_sec": 51.50},
    },
}

def main():
    device = torch.device("cpu")
    print("=" * 65)
    print("  RUNNING E0 & E4 ACROSS SEEDS 45..51 TO UNIFY 10-SEED BENCHMARK")
    print("=" * 65)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        full_results = json.load(f)

    # Insert historical E0 and E4 for seeds 42, 43, 44
    for seed in [42, 43, 44]:
        s_key = str(seed)
        if s_key in full_results:
            full_results[s_key]["E0"] = HISTORICAL_E0_E4[seed]["E0"]
            full_results[s_key]["E4"] = HISTORICAL_E0_E4[seed]["E4"]

    seeds_to_run = [45, 46, 47, 48, 49, 50, 51]
    for seed in seeds_to_run:
        s_key = str(seed)
        print(f"\n>>> Processing Seed {seed} (E0 and E4) <<<")
        t0 = time.perf_counter()

        data_bundle = prepare_experiment_data(
            data_dir="data",
            num_clients=20,
            eval_users_count=30,
            candidates_per_user=50,
            seed=seed,
        )

        # 1. E0 (Centralized baseline)
        print(f"  [Seed {seed}] Running E0: Centralized Baseline...")
        t_e0 = time.perf_counter()
        cfg_e0 = get_e0_centralized_config(seed=seed)
        model_e0, _ = run_centralized_experiment(cfg_e0, data_bundle, device=device)
        eval_e0 = evaluate_model_ranking(
            model=model_e0,
            eval_user_samples=data_bundle.eval_data["eval_user_samples"],
            dataset=data_bundle.full_dataset,
            device=device,
            primary_k=10,
        )
        e0_runtime = time.perf_counter() - t_e0
        full_results[s_key]["E0"] = {
            "ndcg@10": float(eval_e0["ndcg@10"]),
            "recall@10": float(eval_e0["recall@10"]),
            "precision@10": float(eval_e0["precision@10"]),
            "map@10": float(eval_e0["map@10"]),
            "hit_rate@10": float(eval_e0["hit_rate@10"]),
            "train_time_sec": round(e0_runtime, 2),
        }
        print(f"    -> E0 Seed {seed}: NDCG@10={eval_e0['ndcg@10']:.4f}, Recall@10={eval_e0['recall@10']:.4f} in {e0_runtime:.1f}s")

        # 2. E4 (Hybrid DP+HE)
        print(f"  [Seed {seed}] Running E4: Hybrid DP + HE...")
        t_e4 = time.perf_counter()
        cfg_e4 = get_e4_fl_dp_he_config(seed=seed)
        model_e4, _ = run_federated_experiment(cfg_e4, data_bundle, device=device)
        eval_e4 = evaluate_model_ranking(
            model=model_e4,
            eval_user_samples=data_bundle.eval_data["eval_user_samples"],
            dataset=data_bundle.full_dataset,
            device=device,
            primary_k=10,
        )
        e4_runtime = time.perf_counter() - t_e4
        full_results[s_key]["E4"] = {
            "ndcg@10": float(eval_e4["ndcg@10"]),
            "recall@10": float(eval_e4["recall@10"]),
            "precision@10": float(eval_e4["precision@10"]),
            "map@10": float(eval_e4["map@10"]),
            "hit_rate@10": float(eval_e4["hit_rate@10"]),
            "train_time_sec": round(e4_runtime, 2),
        }
        print(f"    -> E4 Seed {seed}: NDCG@10={eval_e4['ndcg@10']:.4f}, Recall@10={eval_e4['recall@10']:.4f} in {e4_runtime:.1f}s")
        print(f"  Seed {seed} complete in {time.perf_counter() - t0:.1f}s")

        # Checkpoint save after every seed
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

    print("\n" + "=" * 65)
    print("  ALL 10 SEEDS COMPLETE FOR ALL 7 CONFIGURATIONS!")
    print("=" * 65)

    # Print summary across all 10 seeds
    print("\nFINAL 10-SEED SUMMARY ACROSS ALL 7 CONFIGURATIONS:")
    for cfg in ["E0", "E1", "E2_DP1", "E2_DP2", "E2_DP3", "E3", "E4"]:
        ndcgs = [full_results[str(s)][cfg]["ndcg@10"] for s in range(42, 52)]
        recs = [full_results[str(s)][cfg]["recall@10"] for s in range(42, 52)]
        precs = [full_results[str(s)][cfg]["precision@10"] for s in range(42, 52)]
        maps = [full_results[str(s)][cfg]["map@10"] for s in range(42, 52)]
        hrs = [full_results[str(s)][cfg]["hit_rate@10"] for s in range(42, 52)]
        print(f"{cfg:8s}: NDCG={np.mean(ndcgs):.4f}+/-{np.std(ndcgs, ddof=1):.4f} | "
              f"Rec={np.mean(recs):.4f}+/-{np.std(recs, ddof=1):.4f} | "
              f"Prec={np.mean(precs):.4f}+/-{np.std(precs, ddof=1):.4f} | "
              f"MAP={np.mean(maps):.4f}+/-{np.std(maps, ddof=1):.4f} | "
              f"HR={np.mean(hrs):.4f}+/-{np.std(hrs, ddof=1):.4f}")

if __name__ == "__main__":
    main()
