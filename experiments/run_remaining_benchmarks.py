#!/usr/bin/env python3
"""
Runs remaining empirical attack and sparsification benchmarks:
1. Membership Inference Attack (MIA) across E1, E2 (DP levels), and E3.
2. Top-K Homomorphic Encryption sparsification sweep across K in [100%, 50%, 20%, 10%, 5%].
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.attacks.mia_attack import evaluate_mia_benchmark
from experiments.federated.run_topk_benchmark import run_sparsification_sweep

def main():
    print("\n=======================================================")
    print("  PHASE 2: EXECUTING MEMBERSHIP INFERENCE ATTACK (MIA)")
    print("=======================================================")
    t0 = time.perf_counter()
    mia_res = evaluate_mia_benchmark(seed=42)
    os.makedirs("experiments/results/attacks", exist_ok=True)
    mia_file = "experiments/results/attacks/mia_benchmark_results.json"
    with open(mia_file, "w", encoding="utf-8") as f:
        json.dump(mia_res, f, indent=2)
    print(f"MIA Benchmark complete in {time.perf_counter() - t0:.1f}s, saved to {mia_file}")

    print("\n=======================================================")
    print("  PHASE 5: EXECUTING TOP-K CKKS SPARSIFICATION BENCHMARK")
    print("=======================================================")
    t1 = time.perf_counter()
    topk_res = run_sparsification_sweep()
    print(f"Top-K Sparsification complete in {time.perf_counter() - t1:.1f}s")

if __name__ == "__main__":
    main()
