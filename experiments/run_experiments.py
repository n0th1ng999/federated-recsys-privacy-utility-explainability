#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import argparse
import copy
import json
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.config import (
    ExperimentConfig,
    get_all_experiment_configs,
    get_quick_experiment_configs,
)
from experiments.data.partitioner import prepare_experiment_data
from experiments.centralized.trainer import run_centralized_experiment
from experiments.federated.fl_runner import run_federated_experiment
from experiments.evaluation.ranking_metrics import evaluate_model_ranking
from experiments.evaluation.performance_metrics import compute_privacy_overhead
from experiments.evaluation.xai_evaluator import (
    evaluate_shap,
    evaluate_lime,
    evaluate_explanation_fidelity,
    evaluate_explanation_stability,
    cosine_similarity,
    spearman_rank_correlation,
    top_k_feature_agreement,
)
from experiments.reporting.table_generator import generate_experiment_tables
from experiments.reporting.plot_generator import generate_experiment_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Privacy, Utility, and Explainability Experiments for Federated Recommender Systems."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Run mode: 'quick' for rapid smoke testing, 'full' for the complete experimental benchmark.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="List of configuration keys to run (e.g. E0 E1 E2_DP2 E3 E4). Default: all configs for the chosen mode.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing raw CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("experiments", "results", "full_benchmark"),
        help="Path to directory where results, tables, and figures are stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip SHAP and LIME evaluations to profile training and ranking only.",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Regenerate tables and figures from existing JSON results without re-running training.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate only figures from existing JSON results without re-running training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    raw_dir = os.path.join(output_dir, "raw")
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    summary_json_path = os.path.join(output_dir, "all_experiments_summary.json")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 75)
    print("  Privacy, Utility & Explainability in Federated Recommenders  ")
    print("=" * 75)
    print(f"Mode: {args.mode.upper()}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")

    # Handle figures-only or reports-only mode
    if args.figures_only or args.reports_only:
        if not os.path.exists(summary_json_path):
            print(f"Error: summary file not found at {summary_json_path}")
            return 1
        with open(summary_json_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        
        if not args.figures_only:
            print("Regenerating tables...")
            generate_experiment_tables(all_results, tables_dir)
            
        print("Regenerating figures (PNG & PDF)...")
        generate_experiment_plots(all_results, figures_dir)
        print(f"Figures successfully updated in: {figures_dir}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Device: {device}")

    # Load configuration suite
    if args.mode == "quick":
        config_suite = get_quick_experiment_configs(seed=args.seed)
    else:
        config_suite = get_all_experiment_configs(seed=args.seed)

    if args.configs:
        selected = {k: v for k, v in config_suite.items() if k in args.configs}
        if not selected:
            print(f"Error: none of the specified configs {args.configs} match available: {list(config_suite.keys())}")
            return 1
        config_suite = selected

    print(f"Scheduled experiments: {list(config_suite.keys())}")

    # 1. Prepare data bundle (controlled variables guarantee exact same data/split)
    first_cfg = next(iter(config_suite.values()))
    print("\nPreparing partitioned dataset and evaluation split...")
    data_t0 = time.perf_counter()
    data_bundle = prepare_experiment_data(
        data_dir=args.data_dir,
        num_clients=first_cfg.fl.num_clients,
        eval_users_count=first_cfg.eval.eval_users,
        candidates_per_user=first_cfg.eval.candidates_per_user,
        seed=args.seed,
        sample_build_workers=1,
    )
    print(f"Data partitioning completed in {time.perf_counter() - data_t0:.2f}s "
          f"({len(data_bundle.client_partitions)} clients, "
          f"{len(data_bundle.eval_data['eval_user_ids'])} eval users, "
          f"{len(data_bundle.full_dataset)} total samples)")

    # Prepare background samples and evaluation samples for XAI
    bg_samples = [data_bundle.full_dataset[i] for i in range(min(20, len(data_bundle.full_dataset)))]
    eval_user_ids = data_bundle.eval_data["eval_user_ids"]
    eval_sample_indices = []
    for uid in eval_user_ids:
        eval_sample_indices.extend(data_bundle.eval_data["eval_user_samples"][uid][:2])

    eval_samples = []
    for idx in eval_sample_indices[:first_cfg.eval.shap_samples]:
        s = data_bundle.full_dataset[idx]
        s["client_id"] = int(data_bundle.full_dataset.sample_client_ids[idx])
        s["product_id"] = int(data_bundle.full_dataset.sample_product_ids[idx])
        eval_samples.append(s)

    all_results: Dict[str, Dict[str, Any]] = {}
    base_fl_attributions: Optional[np.ndarray] = None

    # 2. Main Execution Loop across Configurations
    for exp_key, cfg in config_suite.items():
        print("\n" + "-" * 60)
        print(f"Running Configuration: [{exp_key}] — {cfg.description}")
        print("-" * 60)

        # A. Training
        if not cfg.is_federated:
            model, perf_tracker = run_centralized_experiment(cfg, data_bundle, device)
        else:
            model, perf_tracker = run_federated_experiment(cfg, data_bundle, device)

        print(f"Training completed: {perf_tracker.total_training_time_sec:.2f}s "
              f"(Aggregation: {perf_tracker.aggregation_time_sec:.3f}s, Peak RAM: {perf_tracker.peak_memory_mb:.1f}MB)")

        # B. Recommendation Performance Evaluation
        print("Evaluating recommendation ranking metrics (NDCG, Recall, Precision, MAP, Hit Rate)...")
        eval_t0 = time.perf_counter()
        rec_metrics = evaluate_model_ranking(
            model=model,
            eval_user_samples=data_bundle.eval_data["eval_user_samples"],
            dataset=data_bundle.full_dataset,
            device=device,
            k_values=cfg.eval.k_values,
            primary_k=cfg.eval.primary_k,
        )
        perf_tracker.inference_time_sec = time.perf_counter() - eval_t0
        print(f"Ranking Results: NDCG@10={rec_metrics.get('ndcg@10', 0.0):.4f}, "
              f"Recall@10={rec_metrics.get('recall@10', 0.0):.4f}, "
              f"P@10={rec_metrics.get('precision@10', 0.0):.4f}")

        # C. Explainability Evaluation (SHAP & LIME)
        xai_metrics: Dict[str, Any] = {}
        if not args.skip_xai and len(eval_samples) > 0:
            print("Evaluating explainability (SHAP & LIME)...")
            shap_res = evaluate_shap(
                model=model,
                eval_samples=eval_samples,
                background_samples=bg_samples,
                device=device,
                max_eval_samples=cfg.eval.shap_samples,
            )
            lime_res = evaluate_lime(
                model=model,
                eval_samples=eval_samples,
                background_samples=bg_samples,
                device=device,
                max_eval_samples=cfg.eval.lime_samples,
            )

            # Fidelity evaluation
            print(f"  [XAI - Fidelity] Testing explanation fidelity via top-{cfg.eval.fidelity_top_k} feature perturbation...")
            fid_res = evaluate_explanation_fidelity(
                model=model,
                eval_samples=eval_samples,
                attributions=shap_res["attributions"],
                device=device,
                top_k_perturb=cfg.eval.fidelity_top_k,
            )
            print(f"  [XAI - Fidelity] Mean prediction shift: {fid_res['mean_fidelity_delta']:.4f} (Max Δ: {fid_res['max_fidelity_delta']:.4f})")

            # Stability evaluation
            print(f"  [XAI - Stability] Evaluating repeated-run attribution stability ({cfg.eval.stability_runs} runs)...")
            stab_res = evaluate_explanation_stability(
                model=model,
                sample=eval_samples[0],
                background_samples=bg_samples,
                device=device,
                explainer_type="shap",
                num_runs=cfg.eval.stability_runs,
            )
            print(f"  [XAI - Stability] Stability Cosine Similarity: {stab_res.get('stability_cosine_sim', 1.0):.4f}")

            # Attribution similarity relative to FL baseline (E1) or Centralized (E0)
            if exp_key in ("E1", "E0") and base_fl_attributions is None:
                base_fl_attributions = shap_res["attributions"]

            sim_metrics: Dict[str, float] = {}
            if base_fl_attributions is not None:
                cos_sims = []
                sp_corrs = []
                topk_agrs = []
                for b_vec, curr_vec in zip(base_fl_attributions, shap_res["attributions"]):
                    cos_sims.append(cosine_similarity(b_vec, curr_vec))
                    sp_corrs.append(spearman_rank_correlation(b_vec, curr_vec))
                    topk_agrs.append(top_k_feature_agreement(b_vec, curr_vec, k=5))
                sim_metrics = {
                    "shap_cosine_similarity": float(np.mean(cos_sims)) if cos_sims else 1.0,
                    "shap_spearman_rank_corr": float(np.mean(sp_corrs)) if sp_corrs else 1.0,
                    "shap_top5_agreement": float(np.mean(topk_agrs)) if topk_agrs else 1.0,
                }

            xai_metrics = {
                "shap": {
                    "generation_time_sec": shap_res["generation_time_sec"],
                    "avg_time_per_explanation_sec": shap_res["avg_time_per_explanation_sec"],
                    "num_model_evaluations": shap_res["num_model_evaluations"],
                },
                "lime": {
                    "generation_time_sec": lime_res["generation_time_sec"],
                    "avg_time_per_explanation_sec": lime_res["avg_time_per_explanation_sec"],
                    "num_model_evaluations": lime_res["num_model_evaluations"],
                },
                "fidelity": {
                    "shap_mean_fidelity_delta": fid_res["mean_fidelity_delta"],
                    "shap_max_fidelity_delta": fid_res["max_fidelity_delta"],
                },
                "stability": stab_res,
                "attribution_similarity_to_fl_baseline": sim_metrics,
            }

            # Save structured explanations JSON
            exp_json_path = os.path.join(raw_dir, f"{exp_key}_structured_explanations.json")
            with open(exp_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "config": exp_key,
                    "shap_explanations": shap_res["structured_explanations"],
                    "lime_explanations": lime_res["structured_explanations"],
                    "fidelity_details": fid_res["predictions"],
                }, f, indent=2)

        # Assemble full experiment result
        exp_result = {
            "configuration": exp_key,
            "description": cfg.description,
            "recommendation_metrics": rec_metrics,
            "performance_metrics": perf_tracker.to_dict(),
            "explainability_metrics": xai_metrics,
        }
        all_results[exp_key] = exp_result

        # Save individual experiment raw result
        raw_res_path = os.path.join(raw_dir, f"{exp_key}_results.json")
        with open(raw_res_path, "w", encoding="utf-8") as f:
            json.dump(exp_result, f, indent=2)

    # 3. Save combined summary JSON
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll experimental results saved to {summary_json_path}")

    # 4. Generate Tables and Figures
    print("\nGenerating publication tables and figures...")
    gen_tables = generate_experiment_tables(all_results, tables_dir)
    gen_plots = generate_experiment_plots(all_results, figures_dir)

    print("\n" + "=" * 75)
    print("  EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY  ")
    print("=" * 75)
    print(f"Generated Tables: {list(gen_tables.keys())} in {tables_dir}")
    print(f"Generated Figures: {list(gen_plots.keys())} in {figures_dir}")

    # Print Table 1 to stdout for immediate review
    t1_md_path = gen_tables.get("table1_md")
    if t1_md_path and os.path.exists(t1_md_path):
        print("\n--- Recommendation Performance Summary ---")
        with open(t1_md_path, "r", encoding="utf-8") as f:
            print(f.read())

    t2_md_path = gen_tables.get("table2_md")
    if t2_md_path and os.path.exists(t2_md_path):
        print("\n--- Computational Overhead Summary ---")
        with open(t2_md_path, "r", encoding="utf-8") as f:
            print(f.read())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
