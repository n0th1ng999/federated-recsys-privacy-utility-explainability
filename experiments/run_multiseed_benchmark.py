#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import argparse
import copy
import json
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

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
from experiments.evaluation.xai_evaluator import (
    evaluate_shap,
    evaluate_lime,
    evaluate_explanation_fidelity,
    evaluate_explanation_stability,
    cosine_similarity,
    spearman_rank_correlation,
    top_k_feature_agreement,
)
from experiments.reporting.table_generator import generate_experiment_tables, generate_cross_dataset_table
from experiments.reporting.plot_generator import generate_experiment_plots


def run_single_seed_suite(
    config_suite: Dict[str, ExperimentConfig],
    data_dir: str,
    seed: int,
    skip_xai: bool,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Executes all configurations in config_suite for a given random seed."""
    first_cfg = next(iter(config_suite.values()))
    print(f"\n[Seed {seed}] Preparing partitioned dataset from '{data_dir}'...")
    data_t0 = time.perf_counter()
    data_bundle = prepare_experiment_data(
        data_dir=data_dir,
        num_clients=first_cfg.fl.num_clients,
        eval_users_count=first_cfg.eval.eval_users,
        candidates_per_user=first_cfg.eval.candidates_per_user,
        seed=seed,
        sample_build_workers=1,
    )
    print(f"[Seed {seed}] Dataset ready in {time.perf_counter() - data_t0:.2f}s "
          f"({len(data_bundle.client_partitions)} clients, {len(data_bundle.full_dataset)} samples)")

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

    results: Dict[str, Dict[str, Any]] = {}
    base_fl_attributions: Optional[np.ndarray] = None

    for exp_key, cfg in config_suite.items():
        print(f"\n  >>> [Seed {seed}] Running {exp_key}: {cfg.description}")
        t_start = time.perf_counter()

        if not cfg.is_federated:
            model, perf_tracker = run_centralized_experiment(cfg, data_bundle, device)
        else:
            model, perf_tracker = run_federated_experiment(cfg, data_bundle, device)

        rec_metrics = evaluate_model_ranking(
            model=model,
            eval_user_samples=data_bundle.eval_data["eval_user_samples"],
            dataset=data_bundle.full_dataset,
            device=device,
            k_values=cfg.eval.k_values,
            primary_k=cfg.eval.primary_k,
        )

        xai_metrics: Dict[str, Any] = {}
        if not skip_xai and len(eval_samples) > 0:
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
            fid_res = evaluate_explanation_fidelity(
                model=model,
                eval_samples=eval_samples,
                attributions=shap_res["attributions"],
                device=device,
                top_k_perturb=cfg.eval.fidelity_top_k,
            )
            stab_res = evaluate_explanation_stability(
                model=model,
                sample=eval_samples[0],
                background_samples=bg_samples,
                device=device,
                explainer_type="shap",
                num_runs=cfg.eval.stability_runs,
            )

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

        elapsed = time.perf_counter() - t_start
        print(f"  [Seed {seed}] {exp_key} finished in {elapsed:.2f}s | "
              f"NDCG@10: {rec_metrics.get('ndcg@10', 0.0):.4f}, Recall@10: {rec_metrics.get('recall@10', 0.0):.4f}")

        results[exp_key] = {
            "configuration": exp_key,
            "description": cfg.description,
            "recommendation_metrics": rec_metrics,
            "performance_metrics": perf_tracker.to_dict(),
            "explainability_metrics": xai_metrics,
        }

    return results


def aggregate_seed_results(
    seed_runs: Dict[int, Dict[str, Dict[str, Any]]],
    configs: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Computes mean and sample standard deviation across random seeds for all metrics.
    Stores both mean and f"{key}_std" in the aggregated dictionary.
    """
    aggregated: Dict[str, Dict[str, Any]] = {}
    seeds = sorted(list(seed_runs.keys()))

    def _agg_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if not dict_list:
            return res
        all_keys = set()
        for d in dict_list:
            all_keys.update(d.keys())

        for k in all_keys:
            vals = [d[k] for d in dict_list if k in d and d[k] is not None]
            if not vals:
                continue
            if isinstance(vals[0], (int, float, np.floating, np.integer)):
                numeric_vals = [float(v) for v in vals]
                m = float(np.mean(numeric_vals))
                s = float(np.std(numeric_vals, ddof=1)) if len(numeric_vals) > 1 else 0.0
                res[k] = m
                res[f"{k}_std"] = s
                res[f"{k}_history"] = numeric_vals
            elif isinstance(vals[0], dict):
                res[k] = _agg_dict(vals)
            else:
                res[k] = vals[0]
        return res

    for cfg_key in configs:
        cfg_seed_dicts = [seed_runs[s][cfg_key] for s in seeds if cfg_key in seed_runs[s]]
        if not cfg_seed_dicts:
            continue

        desc = cfg_seed_dicts[0].get("description", "")
        rec_list = [d.get("recommendation_metrics", {}) for d in cfg_seed_dicts]
        perf_list = [d.get("performance_metrics", {}) for d in cfg_seed_dicts]
        xai_list = [d.get("explainability_metrics", {}) for d in cfg_seed_dicts]

        aggregated[cfg_key] = {
            "configuration": cfg_key,
            "description": desc,
            "recommendation_metrics": _agg_dict(rec_list),
            "performance_metrics": _agg_dict(perf_list),
            "explainability_metrics": _agg_dict(xai_list),
            "seeds_evaluated": seeds,
        }

    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed and cross-dataset statistical significance benchmarks."
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--skip-movielens", action="store_true")
    parser.add_argument("--skip-xai", action="store_true")
    parser.add_argument("--output-dir", type=str, default="experiments/results/multiseed_benchmark")
    parser.add_argument("--copy-to-paper", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.abspath(args.output_dir)
    ears_dir = os.path.join(out_dir, "ears")
    ml_dir = os.path.join(out_dir, "movielens")
    tables_dir = os.path.join(out_dir, "tables")
    figures_dir = os.path.join(out_dir, "figures")

    for d in [out_dir, ears_dir, ml_dir, tables_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 80)
    print("  MULTI-SEED & CROSS-DATASET BENCHMARK FOR FEDERATED RECOMMENDER SYSTEMS  ")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Random Seeds: {args.seeds}")
    print(f"Output Directory: {out_dir}")
    print(f"PyTorch Device: {device}")

    # =========================================================================
    # PART 1: EARS Multi-Seed Benchmark
    # =========================================================================
    print("\n" + "#" * 80)
    print("  PHASE 1: EARS Benchmark across Seeds")
    print("#" * 80)

    ears_seed_runs: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for s in args.seeds:
        if args.mode == "quick":
            cfg_suite = get_quick_experiment_configs(seed=s)
        else:
            cfg_suite = get_all_experiment_configs(seed=s)

        if args.configs:
            cfg_suite = {k: v for k, v in cfg_suite.items() if k in args.configs}

        seed_res = run_single_seed_suite(
            config_suite=cfg_suite,
            data_dir="data",
            seed=s,
            skip_xai=args.skip_xai,
            device=device,
        )
        ears_seed_runs[s] = seed_res

        # Save individual seed run
        s_path = os.path.join(ears_dir, f"seed_{s}_results.json")
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump(seed_res, f, indent=2)

    configs_list = list(next(iter(ears_seed_runs.values())).keys())
    ears_aggregated = aggregate_seed_results(ears_seed_runs, configs_list)

    ears_agg_path = os.path.join(ears_dir, "aggregated_summary.json")
    with open(ears_agg_path, "w", encoding="utf-8") as f:
        json.dump(ears_aggregated, f, indent=2)
    print(f"\nEARS aggregated summary saved to {ears_agg_path}")

    # =========================================================================
    # PART 2: MovieLens-100k Validation Benchmark
    # =========================================================================
    ml_aggregated: Dict[str, Dict[str, Any]] = {}
    if not args.skip_movielens:
        print("\n" + "#" * 80)
        print("  PHASE 2: MovieLens-100k Benchmark across Seeds")
        print("#" * 80)

        ml_seed_runs: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for s in args.seeds:
            if args.mode == "quick":
                cfg_suite = get_quick_experiment_configs(seed=s)
            else:
                cfg_suite = get_all_experiment_configs(seed=s)

            if args.configs:
                cfg_suite = {k: v for k, v in cfg_suite.items() if k in args.configs}

            seed_res = run_single_seed_suite(
                config_suite=cfg_suite,
                data_dir="data/movielens",
                seed=s,
                skip_xai=args.skip_xai,
                device=device,
            )
            ml_seed_runs[s] = seed_res

            s_path = os.path.join(ml_dir, f"seed_{s}_results.json")
            with open(s_path, "w", encoding="utf-8") as f:
                json.dump(seed_res, f, indent=2)

        ml_aggregated = aggregate_seed_results(ml_seed_runs, configs_list)
        ml_agg_path = os.path.join(ml_dir, "aggregated_summary.json")
        with open(ml_agg_path, "w", encoding="utf-8") as f:
            json.dump(ml_aggregated, f, indent=2)
        print(f"\nMovieLens aggregated summary saved to {ml_agg_path}")

    # =========================================================================
    # PART 3: Generate Tables and Plots
    # =========================================================================
    print("\n" + "#" * 80)
    print("  PHASE 3: Publication Reporting (Tables & Plots with Error Bars)")
    print("#" * 80)

    gen_tables = generate_experiment_tables(ears_aggregated, tables_dir)
    gen_plots = generate_experiment_plots(ears_aggregated, figures_dir)

    if ml_aggregated:
        t6_files = generate_cross_dataset_table(ears_aggregated, ml_aggregated, tables_dir)
        gen_tables.update(t6_files)

    print(f"Generated Tables: {list(gen_tables.keys())} in {tables_dir}")
    print(f"Generated Figures: {list(gen_plots.keys())} in {figures_dir}")

    # Copy figures to Paper/figures/ if requested
    if args.copy_to_paper:
        paper_fig_dir = os.path.abspath(os.path.join(REPO_ROOT, "Paper", "figures"))
        os.makedirs(paper_fig_dir, exist_ok=True)
        print(f"\nCopying updated publication figures to {paper_fig_dir}...")
        for fname in os.listdir(figures_dir):
            if fname.endswith((".pdf", ".png")):
                src = os.path.join(figures_dir, fname)
                dst = os.path.join(paper_fig_dir, fname)
                shutil.copy2(src, dst)
        print("Paper figures successfully updated!")

    # Print Table 1 & Table 6 to stdout
    t1_md = gen_tables.get("table1_md")
    if t1_md and os.path.exists(t1_md):
        print("\n=== Table 1: Recommendation Performance (Mean ± Std) ===")
        with open(t1_md, "r", encoding="utf-8") as f:
            print(f.read())

    t6_md = gen_tables.get("table6_md")
    if t6_md and os.path.exists(t6_md):
        print("\n=== Table 6: Cross-Dataset Validation (EARS vs. MovieLens) ===")
        with open(t6_md, "r", encoding="utf-8") as f:
            print(f.read())

    print("\n" + "=" * 80)
    print("  MULTI-SEED & CROSS-DATASET BENCHMARK COMPLETE!  ")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
