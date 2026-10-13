#!/usr/bin/env python3
"""
Executes advanced XAI evaluations:
1. Marginal Feature-Permutation Masking (eliminates zero-masking baseline artifact).
2. Faithfulness Deletion AUC (k=1..10 in descending order of feature attribution).
3. Inter-Method Agreement: Spearman rank correlation rho(SHAP, LIME) and Cosine similarity.
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.config import (
    get_e0_centralized_config,
    get_e1_fl_baseline_config,
    get_e2_fl_dp_config,
    get_e3_fl_he_config,
)
from experiments.data.partitioner import prepare_experiment_data
from experiments.centralized.trainer import run_centralized_experiment
from experiments.federated.fl_runner import run_federated_experiment
from experiments.evaluation.xai_evaluator import (
    evaluate_shap,
    evaluate_lime,
    evaluate_permutation_fidelity,
    evaluate_deletion_auc,
    evaluate_inter_method_agreement,
)

OUT_DIR = "experiments/results/xai"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "advanced_xai_benchmark.json")


def main():
    device = torch.device("cpu")
    seed = 42
    print("=" * 65)
    print("  EXECUTING ADVANCED XAI BENCHMARK: FAITHFULNESS & AGREEMENT")
    print("=" * 65)

    data_bundle = prepare_experiment_data(
        data_dir="data",
        num_clients=20,
        eval_users_count=30,
        candidates_per_user=50,
        seed=seed,
    )

    bg_samples = [data_bundle.full_dataset[i] for i in range(min(20, len(data_bundle.full_dataset)))]
    eval_user_ids = data_bundle.eval_data["eval_user_ids"]
    eval_sample_indices = []
    for uid in eval_user_ids:
        eval_sample_indices.extend(data_bundle.eval_data["eval_user_samples"][uid][:2])

    eval_samples = []
    for idx in eval_sample_indices[:20]:
        s = data_bundle.full_dataset[idx]
        s["client_id"] = int(data_bundle.full_dataset.sample_client_ids[idx])
        eval_samples.append(s)

    configs = {
        "E0": get_e0_centralized_config(seed=seed),
        "E1": get_e1_fl_baseline_config(seed=seed),
        "E2_DP2": get_e2_fl_dp_config(level="DP-2", seed=seed),
        "E3": get_e3_fl_he_config(seed=seed),
    }

    # For XAI evaluation, 5 global rounds provide stable model weights quickly
    for c in configs.values():
        if hasattr(c, "fl"):
            c.fl.num_rounds = 5

    xai_results: Dict[str, Any] = {}

    for cfg_name, cfg in configs.items():
        print(f"\n>>> Evaluating {cfg_name} <<<")
        t0 = time.perf_counter()

        if cfg_name == "E0":
            model, _ = run_centralized_experiment(cfg, data_bundle, device=device)
        else:
            model, _ = run_federated_experiment(cfg, data_bundle, device=device)

        print(f"  [{cfg_name}] Computing KernelSHAP attributions (20 instances)...")
        shap_res = evaluate_shap(
            model=model,
            eval_samples=eval_samples,
            background_samples=bg_samples,
            device=device,
            max_eval_samples=20,
        )

        print(f"  [{cfg_name}] Computing LIME attributions (20 instances)...")
        lime_res = evaluate_lime(
            model=model,
            eval_samples=eval_samples,
            background_samples=bg_samples,
            device=device,
            max_eval_samples=20,
        )

        print(f"  [{cfg_name}] Evaluating Permutation Masking Fidelity...")
        perm_fid = evaluate_permutation_fidelity(
            model=model,
            eval_samples=eval_samples,
            background_samples=bg_samples,
            attributions=shap_res["attributions"],
            device=device,
            top_k_perturb=3,
            seed=seed,
        )

        print(f"  [{cfg_name}] Evaluating Faithfulness Deletion AUC (k=1..10)...")
        del_auc = evaluate_deletion_auc(
            model=model,
            eval_samples=eval_samples,
            attributions=shap_res["attributions"],
            device=device,
            max_k=10,
        )

        print(f"  [{cfg_name}] Computing Inter-Method Agreement (SHAP vs LIME)...")
        agreement = evaluate_inter_method_agreement(
            shap_attributions=shap_res["attributions"],
            lime_attributions=lime_res["attributions"],
            k=5,
        )

        xai_results[cfg_name] = {
            "permutation_fidelity_mean_delta": perm_fid["mean_permutation_delta"],
            "deletion_auc_faithfulness": del_auc["mean_deletion_auc"],
            "deletion_auc_std": del_auc["std_deletion_auc"],
            "inter_method_spearman_rho": agreement["inter_method_spearman_rho"],
            "inter_method_cosine_sim": agreement["inter_method_cosine_sim"],
            "inter_method_top5_jaccard": agreement["inter_method_topk_jaccard"],
            "eval_time_sec": round(time.perf_counter() - t0, 2),
        }
        print(f"  -> {cfg_name}: Permutation Δ={perm_fid['mean_permutation_delta']:.4f}, "
              f"Deletion AUC={del_auc['mean_deletion_auc']:.4f}, "
              f"Spearman ρ={agreement['inter_method_spearman_rho']:.4f}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(xai_results, f, indent=2)
    print(f"\nAdvanced XAI Benchmark complete! Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
