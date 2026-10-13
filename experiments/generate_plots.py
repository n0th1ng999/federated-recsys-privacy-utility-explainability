#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.reporting.plot_generator import generate_experiment_plots


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate publication figures (PNG and PDF) from experiment summary JSON."
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=os.path.join("experiments", "results", "full_benchmark", "all_experiments_summary.json"),
        help="Path to all_experiments_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("experiments", "results", "full_benchmark", "figures"),
        help="Path to directory where figures will be saved.",
    )
    args = parser.parse_args()

    summary_path = os.path.abspath(args.summary_json)
    out_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(summary_path):
        # Fallback to quick benchmark if full benchmark is not found
        fallback = os.path.join(REPO_ROOT, "experiments", "results", "all_experiments_summary.json")
        if os.path.exists(fallback):
            summary_path = fallback
            out_dir = os.path.join(os.path.dirname(fallback), "figures")
        else:
            print(f"Error: Summary JSON not found at: {summary_path}")
            return 1

    print(f"Loading experiment results from: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Generating figures into: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    plots = generate_experiment_plots(data, out_dir)

    print("\nSuccessfully generated the following figures:")
    for name, path in plots.items():
        pdf_path = path.replace(".png", ".pdf")
        print(f"  - {os.path.basename(path)} and {os.path.basename(pdf_path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
