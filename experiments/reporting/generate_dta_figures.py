#!/usr/bin/env python3
"""
Generates the consolidated 6 publication figures for Emerald Data Technologies and Applications (DTA):
1. Figure 1: 2-Panel Recommendation Ranking (NDCG@10 and Recall@10 across configurations from 10-seed dataset).
2. Figure 2: 2-Panel Computational Overhead (Training Time & Server Aggregation Latency matching Table V).
3. Figure 3: 3-Panel Explainability Benchmark (SHAP vs LIME Runtime, Masking Sensitivity, Explanation Stability).
4. Figure 4: Multi-Objective Privacy-Utility-Explainability Pareto Trade-off.
5. Figure 5: Empirical Privacy Defense Validation (DLG Gradient Inversion & MIA ROC-AUC).
6. Figure 6: TenSEAL CKKS Top-K Parameter Sparsification Pareto Frontier.
"""
from __future__ import annotations

import os
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "Paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.1


def apply_theme(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.45, color="#c5c5c5", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors="#333333", width=1.1, labelsize=10)


def generate_figure1_ndcg_recall():
    """Generates Figure 1: 2-panel NDCG@10 and Recall@10 across 10 random seeds."""
    configs = ["E0", "E1", "E2_DP1", "E2_DP2", "E2_DP3", "E3", "E4"]
    colors = ["#1b4965", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d", "#6d597a"]

    # 10-seed values from Table IV
    ndcg_means = [0.5048, 0.5379, 0.3675, 0.3929, 0.4199, 0.5377, 0.4109]
    ndcg_stds  = [0.0175, 0.0268, 0.1197, 0.0926, 0.1183, 0.0279, 0.0968]

    recall_means = [0.3501, 0.3851, 0.0962, 0.1216, 0.1436, 0.3860, 0.1587]
    recall_stds  = [0.0281, 0.0158, 0.1275, 0.1002, 0.1208, 0.0161, 0.1061]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300, facecolor="white")

    # Panel (a): NDCG@10
    apply_theme(ax1, grid_axis="y")
    bars1 = ax1.bar(
        configs, ndcg_means, yerr=ndcg_stds, capsize=3.5,
        error_kw=dict(ecolor="#222222", lw=1.1, capthick=1.1),
        color=colors, width=0.52, edgecolor="#222222", lw=1.1, zorder=3
    )
    ax1.set_title("(a) Recommendation Quality (NDCG@10)", fontsize=12.0, fontweight="bold", pad=12)
    ax1.set_ylabel("NDCG@10", fontsize=11.0, fontweight="bold", labelpad=8)
    ax1.set_xlabel("Experimental Configuration", fontsize=11.0, fontweight="bold", labelpad=8)
    ax1.set_ylim(0.0, 0.68)
    for i, bar in enumerate(bars1):
        h = bar.get_height()
        err = ndcg_stds[i]
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h + err), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    # Panel (b): Recall@10
    apply_theme(ax2, grid_axis="y")
    bars2 = ax2.bar(
        configs, recall_means, yerr=recall_stds, capsize=3.5,
        error_kw=dict(ecolor="#222222", lw=1.1, capthick=1.1),
        color=colors, width=0.52, edgecolor="#222222", lw=1.1, zorder=3
    )
    ax2.set_title("(b) Recommendation Coverage (Recall@10)", fontsize=12.0, fontweight="bold", pad=12)
    ax2.set_ylabel("Recall@10", fontsize=11.0, fontweight="bold", labelpad=8)
    ax2.set_xlabel("Experimental Configuration", fontsize=11.0, fontweight="bold", labelpad=8)
    ax2.set_ylim(0.0, 0.49)
    for i, bar in enumerate(bars2):
        h = bar.get_height()
        err = recall_stds[i]
        ax2.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h + err), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    fig.suptitle("Recommendation Ranking Performance across Configurations ($N=10$ random seeds)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, "figure1_ndcg_recall_by_config.pdf")
    out_png = os.path.join(OUTPUT_DIR, "figure1_ndcg_recall_by_config.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 1: {out_pdf}")


def generate_figure2_training_time():
    """Generates Figure 2: 2-panel Computational Overhead (Training Time & Aggregation Latency matching Table V)."""
    configs = ["E0", "E1", "E2_DP1", "E2_DP2", "E2_DP3", "E3", "E4"]

    # Exactly matching Table V
    train_times = [27.32, 25.33, 25.28, 25.45, 26.47, 49.71, 51.15]
    train_stds  = [1.83, 1.71, 1.16, 0.53, 1.96, 0.68, 2.04]

    agg_times = [0.000, 0.042, 0.188, 0.186, 0.196, 24.516, 25.302]
    agg_stds  = [0.000, 0.006, 0.008, 0.004, 0.005, 0.293, 0.458]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300, facecolor="white")

    # Panel (a): Total Training Time
    apply_theme(ax1, grid_axis="y")
    bars1 = ax1.bar(
        configs, train_times, yerr=train_stds, capsize=3.5,
        error_kw=dict(ecolor="#111111", lw=1.1),
        color="#1b4965", width=0.52, edgecolor="#222222", lw=1.1, zorder=3
    )
    ax1.set_title("(a) Total Training Time (15 rounds)", fontsize=12.0, fontweight="bold", pad=12)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=11.0, fontweight="bold", labelpad=8)
    ax1.set_xlabel("Experimental Configuration", fontsize=11.0, fontweight="bold", labelpad=8)
    ax1.set_ylim(0.0, 62.0)
    for i, bar in enumerate(bars1):
        h = bar.get_height()
        err = train_stds[i]
        ax1.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h + err), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    # Panel (b): Server Aggregation Overhead
    apply_theme(ax2, grid_axis="y")
    bars2 = ax2.bar(
        configs, agg_times, yerr=agg_stds, capsize=3.5,
        error_kw=dict(ecolor="#111111", lw=1.1),
        color="#e07a5f", width=0.52, edgecolor="#222222", lw=1.1, zorder=3
    )
    ax2.set_title("(b) Server Aggregation Overhead", fontsize=12.0, fontweight="bold", pad=12)
    ax2.set_ylabel("Aggregation Time (seconds)", fontsize=11.0, fontweight="bold", labelpad=8)
    ax2.set_xlabel("Experimental Configuration", fontsize=11.0, fontweight="bold", labelpad=8)
    ax2.set_ylim(0.0, 31.0)
    for i, bar in enumerate(bars2):
        h = bar.get_height()
        err = agg_stds[i]
        label = f"{h:.3f}s" if h < 1.0 else f"{h:.2f}s"
        ax2.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h + err), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    fig.suptitle("Computational Training and Aggregation Overhead by Configuration", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, "figure2_training_time_breakdown.pdf")
    out_png = os.path.join(OUTPUT_DIR, "figure2_training_time_breakdown.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2: {out_pdf}")


def generate_figure3_xai_runtime_masking_stability():
    """Generates Figure 3: 3-panel Explainability Benchmark."""
    configs = ["E0", "E1", "E2_DP1", "E2_DP2", "E2_DP3", "E3", "E4"]

    # Table VII data
    shap_time = [0.52, 0.63, 0.59, 0.72, 0.69, 0.52, 0.55]
    shap_std  = [0.04, 0.08, 0.04, 0.22, 0.05, 0.02, 0.06]
    lime_time = [3.42, 3.56, 3.48, 3.76, 3.73, 3.41, 3.59]
    lime_std  = [0.06, 0.29, 0.34, 0.02, 0.11, 0.18, 0.18]

    masking = [3.764, 3.996, 0.804, 2.968, 11.701, 3.989, 0.583]
    mask_std = [0.243, 0.675, 0.373, 0.769, 5.660, 0.656, 0.364]

    stability = [0.888, 0.822, 0.787, 0.878, 0.874, 0.685, 0.674]
    stab_std  = [0.092, 0.170, 0.200, 0.106, 0.118, 0.155, 0.246]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=300, facecolor="white")

    # Panel (a): Explanation Runtime
    apply_theme(ax1, grid_axis="y")
    x = np.arange(len(configs))
    w = 0.35
    b1 = ax1.bar(x - w/2, shap_time, w, yerr=shap_std, capsize=3, color="#2a9d8f", edgecolor="#222222", lw=1.0, label="KernelSHAP", zorder=3)
    b2 = ax1.bar(x + w/2, lime_time, w, yerr=lime_std, capsize=3, color="#7209b7", edgecolor="#222222", lw=1.0, label="LIME", zorder=3)
    ax1.set_title("(a) Explainer Latency (s/instance)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0.0, 4.6)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.0)

    # Panel (b): Masking Sensitivity
    apply_theme(ax2, grid_axis="y")
    colors_mask = ["#1b4965", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d", "#6d597a"]
    b_mask = ax2.bar(configs, masking, yerr=mask_std, capsize=3, color=colors_mask, edgecolor="#222222", lw=1.0, zorder=3)
    ax2.set_title("(b) Top-3 Masking Sensitivity $\\Delta_{\\text{mask}}$", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_ylabel("Score Reduction $\\Delta_{\\text{mask}}$", fontsize=10.5, fontweight="bold", labelpad=8)
    ax2.set_ylim(0.0, 19.5)
    for i, bar in enumerate(b_mask):
        h = bar.get_height()
        err = mask_std[i]
        ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h + err), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel (c): Attribution Stability
    apply_theme(ax3, grid_axis="y")
    b_stab = ax3.bar(configs, stability, yerr=stab_std, capsize=3, color=colors_mask, edgecolor="#222222", lw=1.0, zorder=3)
    ax3.set_title("(c) Attribution Stability (Cosine Sim.)", fontsize=11.5, fontweight="bold", pad=12)
    ax3.set_ylabel("Inter-Run Cosine Similarity", fontsize=10.5, fontweight="bold", labelpad=8)
    ax3.set_ylim(0.0, 1.15)
    for i, bar in enumerate(b_stab):
        h = bar.get_height()
        err = stab_std[i]
        ax3.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h + err), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle("Explainability Benchmark: Runtime, Masking Sensitivity, and Stability ($N=3$ seeds)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, "figure3_xai_runtime_masking_stability.pdf")
    out_png = os.path.join(OUTPUT_DIR, "figure3_xai_runtime_masking_stability.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 3: {out_pdf}")


def link_remaining_figures():
    """Links or copies remaining figures so they have unified 1-to-6 numbering."""
    mappings = [
        ("figure6_privacy_utility_tradeoff.pdf", "figure4_privacy_utility_tradeoff.pdf"),
        ("figure6_privacy_utility_tradeoff.png", "figure4_privacy_utility_tradeoff.png"),
        ("figure7_attack_validation_dlg_mia.pdf", "figure5_attack_validation_dlg_mia.pdf"),
        ("figure7_attack_validation_dlg_mia.png", "figure5_attack_validation_dlg_mia.png"),
        ("figure8_topk_sparsification_pareto.pdf", "figure6_topk_sparsification_pareto.pdf"),
        ("figure8_topk_sparsification_pareto.png", "figure6_topk_sparsification_pareto.png"),
    ]
    for src, dst in mappings:
        src_path = os.path.join(OUTPUT_DIR, src)
        dst_path = os.path.join(OUTPUT_DIR, dst)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    generate_figure1_ndcg_recall()
    generate_figure2_training_time()
    generate_figure3_xai_runtime_masking_stability()
    link_remaining_figures()
    print("All consolidated figures successfully generated in Paper/figures/!")
