#!/usr/bin/env python3
"""
Runs expanded seed cohort (seeds 42 to 51) for federated configurations,
focusing on E1 (Plaintext FedAvg), E3 (TenSEAL CKKS), and E2 (DP noise levels).
Saves results and computes formal 10-seed paired statistics, TOST (df=9), and power analysis.
"""
from __future__ import annotations

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss
import json
import time
from typing import Any, Dict, List
import numpy as np
import torch

from experiments.config import (
    get_e0_centralized_config,
    get_e1_fl_baseline_config,
    get_e2_fl_dp_config,
    get_e3_fl_he_config,
    get_e4_fl_dp_he_config,
)
from experiments.data.partitioner import prepare_experiment_data
from experiments.centralized.trainer import run_centralized_experiment
from experiments.federated.fl_runner import run_federated_experiment
from experiments.evaluation.ranking_metrics import evaluate_model_ranking
from experiments.evaluation.statistical_power import (
    compute_paired_stats,
    compute_tost_equivalence,
    compute_tost_power,
    compute_minimum_detectable_effect,
)

OUT_DIR = "experiments/results/10seeds_benchmark"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "ten_seeds_results.json")

# Seeds 42, 43, 44 already evaluated:
HISTORICAL_3SEEDS = {
    42: {
        "E1": {"ndcg@10": 0.5841, "recall@10": 0.3951, "precision@10": 0.8400, "map@10": 0.9412, "hit_rate@10": 0.9000},
        "E3": {"ndcg@10": 0.5862, "recall@10": 0.3955, "precision@10": 0.8400, "map@10": 0.9421, "hit_rate@10": 0.9000},
        "E2_DP1": {"ndcg@10": 0.6053, "recall@10": 0.3541, "precision@10": 0.8200, "map@10": 0.8520, "hit_rate@10": 0.9000},
        "E2_DP2": {"ndcg@10": 0.5173, "recall@10": 0.2458, "precision@10": 0.4900, "map@10": 0.4682, "hit_rate@10": 0.8333},
        "E2_DP3": {"ndcg@10": 0.2861, "recall@10": 0.0075, "precision@10": 0.0200, "map@10": 0.0125, "hit_rate@10": 0.2000},
    },
    43: {
        "E1": {"ndcg@10": 0.5756, "recall@10": 0.4162, "precision@10": 0.8300, "map@10": 0.9301, "hit_rate@10": 0.9000},
        "E3": {"ndcg@10": 0.5782, "recall@10": 0.4180, "precision@10": 0.8333, "map@10": 0.9324, "hit_rate@10": 0.9000},
        "E2_DP1": {"ndcg@10": 0.5463, "recall@10": 0.2875, "precision@10": 0.5333, "map@10": 0.5471, "hit_rate@10": 0.9000},
        "E2_DP2": {"ndcg@10": 0.4201, "recall@10": 0.1415, "precision@10": 0.3400, "map@10": 0.3120, "hit_rate@10": 0.7600},
        "E2_DP3": {"ndcg@10": 0.5583, "recall@10": 0.2481, "precision@10": 0.5600, "map@10": 0.5482, "hit_rate@10": 0.9000},
    },
    44: {
        "E1": {"ndcg@10": 0.5289, "recall@10": 0.4061, "precision@10": 0.8240, "map@10": 0.9202, "hit_rate@10": 0.9000},
        "E3": {"ndcg@10": 0.5293, "recall@10": 0.4065, "precision@10": 0.8247, "map@10": 0.9203, "hit_rate@10": 0.9000},
        "E2_DP1": {"ndcg@10": 0.4147, "recall@10": 0.0658, "precision@10": 0.2467, "map@10": 0.2128, "hit_rate@10": 0.7000},
        "E2_DP2": {"ndcg@10": 0.3529, "recall@10": 0.0965, "precision@10": 0.2500, "map@10": 0.2173, "hit_rate@10": 0.6867},
        "E2_DP3": {"ndcg@10": 0.3622, "recall@10": 0.0681, "precision@10": 0.1420, "map@10": 0.1092, "hit_rate@10": 0.3600},
    },
}


def run_seed(seed: int, device: torch.device) -> Dict[str, Dict[str, float]]:
    print(f"\n=======================================================")
    print(f"  EXECUTING EXPANDED SEED: {seed}")
    print(f"=======================================================")
    t0 = time.perf_counter()
    data_bundle = prepare_experiment_data(
        data_dir="data",
        num_clients=20,
        eval_users_count=30,
        candidates_per_user=50,
        seed=seed,
    )

    results: Dict[str, Dict[str, float]] = {}

    # 1. Plaintext FedAvg (E1)
    print(f"  [Seed {seed}] Running E1: Plaintext FedAvg...")
    cfg_e1 = get_e1_fl_baseline_config(seed=seed)
    model_e1, tracker_e1 = run_federated_experiment(cfg_e1, data_bundle, device=device)
    eval_e1 = evaluate_model_ranking(
        model=model_e1,
        eval_user_samples=data_bundle.eval_data["eval_user_samples"],
        dataset=data_bundle.full_dataset,
        device=device,
        primary_k=10,
    )
    results["E1"] = {
        "ndcg@10": eval_e1["ndcg@10"],
        "recall@10": eval_e1["recall@10"],
        "precision@10": eval_e1["precision@10"],
        "map@10": eval_e1["map@10"],
        "hit_rate@10": eval_e1["hit_rate@10"],
        "train_time_sec": tracker_e1.total_training_time_sec,
    }
    print(f"    -> E1 Result: NDCG@10 = {eval_e1['ndcg@10']:.4f}, Recall@10 = {eval_e1['recall@10']:.4f}")

    # 2. TenSEAL CKKS HE (E3)
    print(f"  [Seed {seed}] Running E3: TenSEAL CKKS HE...")
    cfg_e3 = get_e3_fl_he_config(seed=seed)
    model_e3, tracker_e3 = run_federated_experiment(cfg_e3, data_bundle, device=device)
    eval_e3 = evaluate_model_ranking(
        model=model_e3,
        eval_user_samples=data_bundle.eval_data["eval_user_samples"],
        dataset=data_bundle.full_dataset,
        device=device,
        primary_k=10,
    )
    diff = eval_e3["ndcg@10"] - eval_e1["ndcg@10"]
    results["E3"] = {
        "ndcg@10": eval_e3["ndcg@10"],
        "recall@10": eval_e3["recall@10"],
        "precision@10": eval_e3["precision@10"],
        "map@10": eval_e3["map@10"],
        "hit_rate@10": eval_e3["hit_rate@10"],
        "train_time_sec": tracker_e3.total_training_time_sec,
        "paired_diff_vs_e1": diff,
    }
    print(f"    -> E3 Result: NDCG@10 = {eval_e3['ndcg@10']:.4f}, Paired diff = {diff:+.5f}")

    # 3. Differential Privacy (E2: DP1, DP2, DP3)
    for lvl in ["DP-1", "DP-2", "DP-3"]:
        cfg_dp = get_e2_fl_dp_config(level=lvl, seed=seed)
        model_dp, tracker_dp = run_federated_experiment(cfg_dp, data_bundle, device=device)
        eval_dp = evaluate_model_ranking(
            model=model_dp,
            eval_user_samples=data_bundle.eval_data["eval_user_samples"],
            dataset=data_bundle.full_dataset,
            device=device,
            primary_k=10,
        )
        tag = f"E2_{lvl.replace('-', '')}"
        results[tag] = {
            "ndcg@10": eval_dp["ndcg@10"],
            "recall@10": eval_dp["recall@10"],
            "precision@10": eval_dp["precision@10"],
            "map@10": eval_dp["map@10"],
            "hit_rate@10": eval_dp["hit_rate@10"],
            "train_time_sec": tracker_dp.total_training_time_sec,
        }
        print(f"    -> {tag} (sigma={cfg_dp.dp.noise_multiplier}): NDCG@10 = {eval_dp['ndcg@10']:.4f}")

    print(f"  [Seed {seed}] Completed in {time.perf_counter() - t0:.1f}s")
    return results


def main():
    device = torch.device("cpu")
    all_results = dict(HISTORICAL_3SEEDS)

    seeds_to_run = [45, 46, 47, 48, 49, 50, 51]
    print(f"Executing seeds: {seeds_to_run}")

    for s in seeds_to_run:
        res = run_seed(seed=s, device=device)
        all_results[s] = res

        # Checkpoint save
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("  ALL 10 SEEDS COMPLETE! COMPUTING FORMAL STATISTICAL TESTS")
    print("=" * 80)

    e1_ndcg = [all_results[s]["E1"]["ndcg@10"] for s in sorted(all_results.keys())]
    e3_ndcg = [all_results[s]["E3"]["ndcg@10"] for s in sorted(all_results.keys())]

    paired_stats = compute_paired_stats(e1_ndcg, e3_ndcg, alpha=0.05)
    tost_001 = compute_tost_equivalence(e1_ndcg, e3_ndcg, margin=0.01, alpha=0.05)
    tost_0005 = compute_tost_equivalence(e1_ndcg, e3_ndcg, margin=0.005, alpha=0.05)
    mde_80 = compute_minimum_detectable_effect(n=10, sd=paired_stats["std_diff"], power=0.80)

    summary = {
        "seeds": sorted(list(all_results.keys())),
        "n": len(all_results),
        "e1_ndcg_mean": float(np.mean(e1_ndcg)),
        "e1_ndcg_std": float(np.std(e1_ndcg, ddof=1)),
        "e3_ndcg_mean": float(np.mean(e3_ndcg)),
        "e3_ndcg_std": float(np.std(e3_ndcg, ddof=1)),
        "paired_stats": paired_stats,
        "tost_margin_0_01": tost_001,
        "tost_margin_0_005": tost_0005,
        "minimum_detectable_effect_80pct": mde_80,
    }

    print("\n10-Seed Statistical Summary for H1 (E1 vs E3):")
    print(f"  E1 NDCG@10: {summary['e1_ndcg_mean']:.4f} +/- {summary['e1_ndcg_std']:.4f}")
    print(f"  E3 NDCG@10: {summary['e3_ndcg_mean']:.4f} +/- {summary['e3_ndcg_std']:.4f}")
    print(f"  Paired diff: {paired_stats['mean_diff']:+.5f} +/- {paired_stats['std_diff']:.5f}")
    print(f"  95% CI: [{paired_stats['ci_lower']:+.5f}, {paired_stats['ci_upper']:+.5f}]")
    print(f"  Cohen's d: {paired_stats['cohens_d']:.4f}")
    print(f"  TOST (margin +/-0.01): t1={tost_001['t1']:.2f}, t2={tost_001['t2']:.2f}, p_TOST={tost_001['p_tost']:.4e}, Equivalent={tost_001['is_equivalent']}, Power={tost_001['observed_power']:.4f}")
    print(f"  TOST (margin +/-0.005): t1={tost_0005['t1']:.2f}, t2={tost_0005['t2']:.2f}, p_TOST={tost_0005['p_tost']:.4e}, Equivalent={tost_0005['is_equivalent']}")
    print(f"  Minimum Detectable Effect (80% power): {mde_80:.6f}")

    final_path = os.path.join(OUT_DIR, "ten_seeds_statistical_summary.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary successfully saved to {final_path}")


if __name__ == "__main__":
    main()
