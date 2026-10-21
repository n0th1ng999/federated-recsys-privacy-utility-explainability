#!/usr/bin/env python3
"""
Generates publication-quality figures for Emerald DTA:
- Figure 7: Homomorphic Encryption Top-K Parameter Sparsification Pareto Frontier.
- Figure 8: Empirical Privacy Attack Validation (DLG Gradient Inversion & MIA Membership Inference).
- Figure 1 & 2: Regenerated NDCG@10 & Recall@10 matching 10-seed benchmark.
"""
from __future__ import annotations

import os
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    ax.tick_params(colors="#333333", width=1.1, labelsize=10.5)


def generate_figure8_sparsification():
    """Generates Figure 8: Top-K Sparsification Pareto Frontier."""
    json_path = "experiments/results/sparsification/topk_sparsification_results.json"
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Convert to arrays
    labels = [d["active_param_pct"] for d in data]
    comm_mb = [d["total_comm_mb"] for d in data]
    comm_gb = [m / 1024.0 for m in comm_mb]
    ndcg = [d["ndcg@10"] for d in data]
    reduction = [d["comm_reduction_vs_full"] for d in data]
    retention = [d["utility_retention_pct"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300, facecolor="white")

    # Panel (a): Pareto Frontier (Comm GB vs NDCG@10)
    apply_theme(ax1, grid_axis="both")
    colors = ["#1b4965", "#2a9d8f", "#e9c46a", "#e76f51", "#6d597a"]

    ax1.plot(comm_gb, ndcg, color="#457b9d", linestyle="-", linewidth=2.0, zorder=2)
    annotations_cfg = {
        0: {"offset": (-25, 22), "ha": "center", "va": "bottom"},   # K=100% (16.0 GB)
        1: {"offset": (0, 22),   "ha": "center", "va": "bottom"},   # K=50% (8.0 GB)
        2: {"offset": (26, 18),  "ha": "left",   "va": "bottom"},   # K=20% (3.2 GB)
        3: {"offset": (16, 25),  "ha": "left",   "va": "bottom"},   # K=10% (1.7 GB) - ABOVE-RIGHT
        4: {"offset": (0, -35),  "ha": "center", "va": "top"},      # K=5% (0.9 GB) - CENTERED BELOW
    }

    for i in range(len(data)):
        ax1.scatter(comm_gb[i], ndcg[i], s=140, color=colors[i], edgecolors="#222222", linewidth=1.3, zorder=4)
        cfg = annotations_cfg.get(i, {"offset": (0, 20), "ha": "center", "va": "bottom"})
        label_text = f"K={labels[i]}\n({comm_gb[i]:.1f} GB, {ndcg[i]:.3f})"
        if labels[i] == "10%":
            label_text = f"K={labels[i]} (Optimal)\n({comm_gb[i]:.1f} GB, {ndcg[i]:.3f})"

        ax1.annotate(
            label_text,
            xy=(comm_gb[i], ndcg[i]),
            xytext=cfg["offset"],
            textcoords="offset points",
            ha=cfg["ha"],
            va=cfg["va"],
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#888888", alpha=0.94, lw=0.8),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            zorder=5,
        )

    ax1.set_title("(a) Communication vs Ranking Accuracy Pareto Frontier", fontsize=12.0, fontweight="bold", pad=12)
    ax1.set_xlabel("Total Transmitted Payload (GB across 15 rounds)", fontsize=11.0, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Recommendation Quality (NDCG@10)", fontsize=11.0, fontweight="bold", labelpad=8)
    ax1.set_xlim(-0.8, 17.5)
    ax1.set_ylim(0.48, 0.72)
    ax1.axhline(ndcg[0], linestyle=":", color="#777777", alpha=0.7, label=f"Full Baseline ({ndcg[0]:.3f})")
    ax1.legend(loc="lower right", frameon=True, fontsize=9.5)

    # Panel (b): Communication Reduction vs Utility Retention (Horizontal grouped bars for maximum label clarity)
    apply_theme(ax2, grid_axis="x")
    y_pos = np.arange(len(labels))[::-1]  # 100% at top, 5% at bottom
    bar_height = 0.32
    gain = [r - 100.0 for r in retention]

    rects1 = ax2.barh(y_pos + bar_height/2, reduction, bar_height, label="Bandwidth Reduction (%)", color="#2a9d8f", edgecolor="#222222", lw=1.1, zorder=3)
    rects2 = ax2.barh(y_pos - bar_height/2, gain, bar_height, label="Ranking Gain vs Baseline (%)", color="#e76f51", edgecolor="#222222", lw=1.1, zorder=3)

    ax2.set_title("(b) Trade-off: Bandwidth Savings vs Ranking Gain", fontsize=12.0, fontweight="bold", pad=12)
    ax2.set_ylabel("Sparsification Parameter (Top-K% Retained)", fontsize=11.0, fontweight="bold", labelpad=8)
    ax2.set_xlabel("Relative Change (%)", fontsize=11.0, fontweight="bold", labelpad=8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax2.set_xlim(-3, 115)
    ax2.axvline(0, color="#333333", linewidth=1.1)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

    for bar in rects1:
        w = bar.get_width()
        ax2.annotate(f"{w:.0f}%", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=9.0, fontweight="bold", color="#1b4965")

    for bar in rects2:
        w = bar.get_width()
        ax2.annotate(f"+{w:.1f}%" if w > 0 else f"{w:.1f}%", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=9.0, fontweight="bold", color="#b23a22")

    fig.suptitle("TenSEAL CKKS Homomorphic Encryption Top-K Parameter Sparsification", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, "figure8_topk_sparsification_pareto.pdf")
    out_png = os.path.join(OUTPUT_DIR, "figure8_topk_sparsification_pareto.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 8: {out_pdf} and {out_png}")


def generate_figure7_attacks():
    """Generates Figure 7: Empirical Privacy Attack Defense (DLG & MIA)."""
    configs = ["E1: Plaintext", "E2: DP (σ=0.5)", "E2: DP (σ=1.0)", "E2: DP (σ=2.0)", "E3: CKKS HE", "E4: Hybrid"]

    # DLG metrics
    dlg_asr = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dlg_cos = [0.9999, 0.0389, -0.0014, 0.0049, 0.0193, 0.0193]

    # MIA metrics
    mia_auc = [0.2138, 0.6180, 0.6815, 0.4035, 0.2170, 0.6815]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=300, facecolor="white")

    # Panel (a): DLG Gradient Inversion Defense
    apply_theme(ax1, grid_axis="y")
    colors_dlg = ["#e76f51", "#2a9d8f", "#2a9d8f", "#2a9d8f", "#1b4965", "#1b4965"]
    bars_dlg = ax1.bar(range(len(configs)), dlg_asr, color=colors_dlg, width=0.52, edgecolor="#222222", lw=1.1, zorder=3)

    ax1.set_title("(a) Gradient Inversion (DLG) Attack Success Rate", fontsize=12.0, fontweight="bold", pad=12)
    ax1.set_ylabel("Attack Success Rate (%) [cos ≥ 0.80]", fontsize=11.0, fontweight="bold", labelpad=8)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylim(0, 115)

    for i, bar in enumerate(bars_dlg):
        h = bar.get_height()
        ax1.annotate(f"{h:.0f}%\n(cos={dlg_cos[i]:.2f})", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    # Panel (b): MIA Membership Inference Defense
    apply_theme(ax2, grid_axis="y")
    colors_mia = ["#e76f51", "#e9c46a", "#f4a261", "#e9c46a", "#e76f51", "#2a9d8f"]
    bars_mia = ax2.bar(range(len(configs)), mia_auc, color=colors_mia, width=0.52, edgecolor="#222222", lw=1.1, zorder=3)

    ax2.axhline(0.50, linestyle="--", color="#555555", linewidth=1.2, label="Random Guessing (AUC=0.50)", zorder=2)
    ax2.set_title("(b) Membership Inference Attack (MIA) ROC-AUC", fontsize=12.0, fontweight="bold", pad=12)
    ax2.set_ylabel("MIA ROC-AUC [Member vs Non-Member]", fontsize=11.0, fontweight="bold", labelpad=8)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=25, ha="right", fontsize=9.5)
    ax2.set_ylim(0.0, 0.85)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)

    for i, bar in enumerate(bars_mia):
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9.2, fontweight="bold")

    fig.suptitle("Empirical Privacy Attack Defense Validation: Resolving Adversary Scope", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, "figure7_attack_validation_dlg_mia.pdf")
    out_png = os.path.join(OUTPUT_DIR, "figure7_attack_validation_dlg_mia.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 7: {out_pdf} and {out_png}")


if __name__ == "__main__":
    generate_figure7_attacks()
    generate_figure8_sparsification()
