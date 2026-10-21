#!/usr/bin/env python3
"""
================================================================================
  Paper Reproduction Master CLI:
  Privacy, Utility, and Explainability Trade-offs in Privacy-Preserving
  Federated Recommender Systems
================================================================================
Target Journal: Data Technologies and Applications (Emerald DTA)

This script provides a unified entrypoint to reproduce all experimental findings,
benchmarks, privacy attacks, sparsification sweeps, and figures reported in the paper.

Usage:
  python reproduce.py --mode smoke           # 60s fast sanity check across configs
  python reproduce.py --mode figures         # Regenerate Figures 1 through 8
  python reproduce.py --mode tables          # Regenerate Tables I through XII
  python reproduce.py --mode attacks         # Run MIA & DLG empirical privacy attacks
  python reproduce.py --mode sparsification  # Run TenSEAL Top-K sparsification benchmark
  python reproduce.py --mode xai             # Run advanced XAI faithfulness & agreement
  python reproduce.py --mode validate        # Validate LaTeX citations and label consistency
  python reproduce.py --mode all             # Run validation, figures, and benchmark suites
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import subprocess

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def run_smoke_test() -> int:
    banner("Running Fast Smoke Test (1-Seed, Micro-Rounds)")
    print("Verifying PyTorch model architecture, Federated FedAvg, Differential Privacy,")
    print("TenSEAL CKKS Homomorphic Encryption, and Ranking Evaluation...\n")
    
    cmd = [
        sys.executable,
        "-m", "experiments.run_experiments",
        "--mode", "quick",
        "--configs", "E0", "E1", "E2_DP1", "E3",
        "--output-dir", "experiments/results/smoke_test",
        "--skip-xai",
    ]
    t0 = time.perf_counter()
    ret = subprocess.call(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    if ret == 0:
        print(f"\n[OK] Smoke test completed successfully in {elapsed:.1f}s.")
    else:
        print(f"\n[FAIL] Smoke test exited with error code {ret}.")
    return ret


def run_figures() -> int:
    banner("Regenerating Manuscript Figures 1 through 8")
    t0 = time.perf_counter()
    
    # 1. Figures 1 through 6
    print("Generating Figures 1 to 6 via plot_generator...")
    from experiments.reporting.plot_generator import generate_plots_from_10seeds
    generate_plots_from_10seeds(
        ten_seeds_json=os.path.join(PROJECT_ROOT, "experiments/results/10seeds_benchmark/ten_seeds_results.json"),
        multiseed_summary_json=os.path.join(PROJECT_ROOT, "experiments/results/multiseed_benchmark/ears/aggregated_summary.json"),
        output_dir=os.path.join(PROJECT_ROOT, "Paper/figures"),
    )

    # 2. Figures 7 and 8
    print("Generating Figures 7 (MIA/DLG Defense) & 8 (Top-K Pareto) via generate_new_figures...")
    from experiments.reporting.generate_new_figures import (
        generate_figure7_attacks,
        generate_figure8_sparsification,
    )
    generate_figure7_attacks()
    generate_figure8_sparsification()

    elapsed = time.perf_counter() - t0
    print(f"\n[OK] All Figures 1-8 generated in Paper/figures/ in {elapsed:.1f}s.")
    return 0


def run_tables() -> int:
    banner("Regenerating Manuscript Tables I through XII")
    t0 = time.perf_counter()
    
    cmd = [
        sys.executable,
        "-c",
        "import json; from experiments.reporting.table_generator import generate_experiment_tables, generate_cross_dataset_table; "
        "summary = json.load(open('experiments/results/multiseed_benchmark/ears/aggregated_summary.json')); "
        "tables = generate_experiment_tables(summary, 'experiments/results/multiseed_benchmark/tables'); "
        "print('Generated tables:', list(tables.keys()))"
    ]
    ret = subprocess.call(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    if ret == 0:
        print(f"\n[OK] Tables successfully generated in {elapsed:.1f}s.")
    else:
        print(f"\n[FAIL] Table generation failed with error code {ret}.")
    return ret


def run_attacks() -> int:
    banner("Executing Empirical Privacy Defense Benchmark (MIA & DLG Inversion)")
    t0 = time.perf_counter()
    from experiments.attacks.mia_attack import evaluate_mia_benchmark
    import json

    res = evaluate_mia_benchmark(seed=42)
    out_dir = os.path.join(PROJECT_ROOT, "experiments/results/attacks")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "mia_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    elapsed = time.perf_counter() - t0
    print(f"\n[OK] Privacy attack evaluation completed in {elapsed:.1f}s, saved to {out_file}.")
    return 0


def run_sparsification() -> int:
    banner("Executing Top-K CKKS Parameter Sparsification Sweep")
    t0 = time.perf_counter()
    from experiments.federated.run_topk_benchmark import run_sparsification_sweep

    res = run_sparsification_sweep()
    elapsed = time.perf_counter() - t0
    print(f"\n[OK] Top-K sparsification benchmark completed in {elapsed:.1f}s.")
    return 0


def run_xai() -> int:
    banner("Executing Advanced XAI Benchmark (Marginal Permutation & Deletion AUC)")
    t0 = time.perf_counter()
    cmd = [sys.executable, "experiments/evaluation/run_advanced_xai_benchmark.py"]
    ret = subprocess.call(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    if ret == 0:
        print(f"\n[OK] Advanced XAI benchmark completed in {elapsed:.1f}s.")
    else:
        print(f"\n[FAIL] Advanced XAI benchmark exited with code {ret}.")
    return ret


def run_validate() -> int:
    banner("Validating LaTeX Manuscript Integrity")
    cmd = [sys.executable, "tooling/validate_manuscript.py"]
    ret = subprocess.call(cmd, cwd=PROJECT_ROOT)
    return ret


def main():
    parser = argparse.ArgumentParser(
        description="Master reproduction CLI for Emerald DTA Federated Recommender System Paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "figures", "tables", "attacks", "sparsification", "xai", "validate", "all"],
        default="figures",
        help="Reproduction mode to execute (default: figures)",
    )

    args = parser.parse_args()

    dispatch = {
        "smoke": run_smoke_test,
        "figures": run_figures,
        "tables": run_tables,
        "attacks": run_attacks,
        "sparsification": run_sparsification,
        "xai": run_xai,
        "validate": run_validate,
    }

    if args.mode == "all":
        code = run_validate()
        if code == 0:
            code = run_figures()
        if code == 0:
            code = run_tables()
        if code == 0:
            code = run_attacks()
        if code == 0:
            code = run_sparsification()
        sys.exit(code)
    else:
        code = dispatch[args.mode]()
        sys.exit(code)


if __name__ == "__main__":
    main()
