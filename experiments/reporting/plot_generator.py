from __future__ import annotations

import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _extract_metric(d: Dict[str, Any], key: str, default: float = 0.0) -> tuple[float, float]:
    """Extracts (mean, std) from a metric dictionary."""
    val = d.get(key, default)
    if isinstance(val, dict) and "mean" in val:
        return float(val["mean"]), float(val.get("std", 0.0))
    std_val = d.get(f"{key}_std", 0.0)
    try:
        return float(val), float(std_val)
    except (ValueError, TypeError):
        return default, 0.0


def _apply_publication_theme(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Applies a clean, high-readability IEEE/ACM publication theme to an axis."""
    ax.set_facecolor("white")
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.45, color="#c5c5c5", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors="#333333", width=1.1, labelsize=11)


def _get_config_color(config_name: str) -> str:
    """Returns consistent, publication-grade academic color for each configuration."""
    color_map = {
        "E0": "#1b4965",       # Deep Navy (Centralized)
        "E1": "#2a9d8f",       # Teal (FL Baseline)
        "E2_DP1": "#e9c46a",   # Soft Gold (DP Low)
        "E2_DP2": "#f4a261",   # Ochre (DP Mid)
        "E2_DP3": "#e76f51",   # Coral (DP High)
        "E3": "#457b9d",       # Slate Blue (HE CKKS)
        "E4": "#6d597a",       # Plum Slate (DP+HE)
    }
    return color_map.get(config_name, "#5c6b73")


def generate_experiment_plots(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generates publication-quality figures (PDF and high-resolution PNG):
    1. NDCG@10 by configuration (annotated values, error bars if available)
    2. Recall@10 by configuration (annotated values, error bars if available)
    3. Computational Overhead: 2-panel breakdown of Training Time & Aggregation Time
    4. SHAP vs. LIME execution runtime with labeled values & error bars
    5. 2-panel comparison of Explanation Fidelity Δ and Stability
    6. Privacy–Utility–Explainability Pareto trade-off scatter plot with unclipped callouts
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = {}

    configs = list(all_results.keys())
    config_colors = [_get_config_color(c) for c in configs]

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.1

    # -------------------------------------------------------------
    # Figure 1: NDCG@10 by Configuration
    # -------------------------------------------------------------
    ndcg_data = [_extract_metric(all_results[c].get("recommendation_metrics", {}), "ndcg@10", 0.0) for c in configs]
    ndcg_vals = [x[0] for x in ndcg_data]
    ndcg_stds = [x[1] for x in ndcg_data]
    has_ndcg_err = any(s > 0 for s in ndcg_stds)
    yerr_ndcg = ndcg_stds if has_ndcg_err else None

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=300, facecolor="white")
    _apply_publication_theme(ax, grid_axis="y")

    bars = ax.bar(
        configs,
        ndcg_vals,
        yerr=yerr_ndcg,
        capsize=4 if has_ndcg_err else 0,
        error_kw=dict(ecolor="#222222", lw=1.2, capthick=1.2),
        color=config_colors,
        width=0.52,
        edgecolor="#222222",
        linewidth=1.1,
        zorder=3,
    )
    ax.set_title("Recommendation Performance (NDCG@10) across Privacy Configurations", fontsize=14, fontweight="bold", pad=15, color="#111111")
    ax.set_ylabel("NDCG@10", fontsize=12.5, fontweight="bold", color="#222222")
    ax.set_xlabel("Experimental Configuration", fontsize=12.5, fontweight="bold", labelpad=9, color="#222222")
    max_ndcg = max(ndcg_vals + [0.1])
    ax.set_ylim(0.0, max_ndcg * 1.25)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        err_offset = ndcg_stds[i] if has_ndcg_err else 0.0
        label_text = f"{h:.4f}" if not has_ndcg_err else f"{h:.4f}"
        ax.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, h + err_offset),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#222222",
        )

    plt.tight_layout()
    f1_png = os.path.join(output_dir, "figure1_ndcg_by_config.png")
    f1_pdf = os.path.join(output_dir, "figure1_ndcg_by_config.pdf")
    plt.savefig(f1_png, bbox_inches="tight")
    plt.savefig(f1_pdf, bbox_inches="tight")
    plt.close()
    generated_plots["figure1_ndcg"] = f1_png

    # -------------------------------------------------------------
    # Figure 2: Recall@10 by Configuration
    # -------------------------------------------------------------
    recall_data = [_extract_metric(all_results[c].get("recommendation_metrics", {}), "recall@10", 0.0) for c in configs]
    recall_vals = [x[0] for x in recall_data]
    recall_stds = [x[1] for x in recall_data]
    has_rec_err = any(s > 0 for s in recall_stds)
    yerr_rec = recall_stds if has_rec_err else None

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=300, facecolor="white")
    _apply_publication_theme(ax, grid_axis="y")

    bars = ax.bar(
        configs,
        recall_vals,
        yerr=yerr_rec,
        capsize=4 if has_rec_err else 0,
        error_kw=dict(ecolor="#222222", lw=1.2, capthick=1.2),
        color=config_colors,
        width=0.52,
        edgecolor="#222222",
        linewidth=1.1,
        zorder=3,
    )
    ax.set_title("Recommendation Recall@10 across Privacy Configurations", fontsize=14, fontweight="bold", pad=15, color="#111111")
    ax.set_ylabel("Recall@10", fontsize=12.5, fontweight="bold", color="#222222")
    ax.set_xlabel("Experimental Configuration", fontsize=12.5, fontweight="bold", labelpad=9, color="#222222")
    max_recall = max(recall_vals + [0.1])
    ax.set_ylim(0.0, max_recall * 1.25)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        err_offset = recall_stds[i] if has_rec_err else 0.0
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h + err_offset),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#222222",
        )

    plt.tight_layout()
    f2_png = os.path.join(output_dir, "figure2_recall_by_config.png")
    f2_pdf = os.path.join(output_dir, "figure2_recall_by_config.pdf")
    plt.savefig(f2_png, bbox_inches="tight")
    plt.savefig(f2_pdf, bbox_inches="tight")
    plt.close()
    generated_plots["figure2_recall"] = f2_png

    # -------------------------------------------------------------
    # Figure 3: Computational Overhead Breakdown (2-Panel)
    # -------------------------------------------------------------
    train_data = [_extract_metric(all_results[c].get("performance_metrics", {}), "total_training_time_sec", 0.0) for c in configs]
    agg_data = []
    for c in configs:
        perf = all_results[c].get("performance_metrics", {})
        if "aggregation_time_sec" in perf:
            agg_data.append(_extract_metric(perf, "aggregation_time_sec", 0.0))
        elif "server_aggregation_time_sec" in perf:
            agg_data.append(_extract_metric(perf, "server_aggregation_time_sec", 0.0))
        else:
            agg_data.append((0.0, 0.0))

    train_times = [x[0] for x in train_data]
    train_stds = [x[1] for x in train_data]
    agg_times = [x[0] for x in agg_data]
    agg_stds = [x[1] for x in agg_data]

    has_train_err = any(s > 0 for s in train_stds)
    has_agg_err = any(s > 0 for s in agg_stds)

    fig, (ax_train, ax_agg) = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=300, facecolor="white")

    # Subplot 3A: Total Training Time
    _apply_publication_theme(ax_train, grid_axis="y")
    bars_t = ax_train.bar(
        configs,
        train_times,
        yerr=train_stds if has_train_err else None,
        capsize=3.5 if has_train_err else 0,
        error_kw=dict(ecolor="#111111", lw=1.1),
        color="#1b4965",
        width=0.52,
        edgecolor="#222222",
        linewidth=1.1,
        zorder=3,
    )
    ax_train.set_title("(a) Total Training Time", fontsize=12.5, fontweight="bold", pad=13)
    ax_train.set_ylabel("Time (seconds)", fontsize=11.5, fontweight="bold")
    ax_train.set_xlabel("Configuration", fontsize=11.5, fontweight="bold", labelpad=8)
    max_train_val = max([train_times[i] + (train_stds[i] if has_train_err else 0.0) for i in range(len(train_times))] + [1.0])
    ax_train.set_ylim(0.0, max_train_val * 1.25)
    for i, bar in enumerate(bars_t):
        h = bar.get_height()
        err = train_stds[i] if has_train_err else 0.0
        ax_train.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h + err), xytext=(0, 5),
                          textcoords="offset points", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # Subplot 3B: Server Aggregation Time
    _apply_publication_theme(ax_agg, grid_axis="y")
    bars_a = ax_agg.bar(
        configs,
        agg_times,
        yerr=agg_stds if has_agg_err else None,
        capsize=3.5 if has_agg_err else 0,
        error_kw=dict(ecolor="#111111", lw=1.1),
        color="#e07a5f",
        width=0.52,
        edgecolor="#222222",
        linewidth=1.1,
        zorder=3,
    )
    ax_agg.set_title("(b) Server Aggregation Overhead", fontsize=12.5, fontweight="bold", pad=13)
    ax_agg.set_ylabel("Time (seconds)", fontsize=11.5, fontweight="bold")
    ax_agg.set_xlabel("Configuration", fontsize=11.5, fontweight="bold", labelpad=8)
    max_agg_val = max([agg_times[i] + (agg_stds[i] if has_agg_err else 0.0) for i in range(len(agg_times))] + [0.1])
    ax_agg.set_ylim(0.0, max_agg_val * 1.28)
    for i, bar in enumerate(bars_a):
        h = bar.get_height()
        err = agg_stds[i] if has_agg_err else 0.0
        label = f"{h:.3f}s" if h < 1.0 else f"{h:.2f}s"
        ax_agg.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, h + err), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    fig.suptitle("Computational Training and Aggregation Overhead by Configuration", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    f3_png = os.path.join(output_dir, "figure3_training_time_breakdown.png")
    f3_pdf = os.path.join(output_dir, "figure3_training_time_breakdown.pdf")
    plt.savefig(f3_png, bbox_inches="tight")
    plt.savefig(f3_pdf, bbox_inches="tight")
    plt.close()
    generated_plots["figure3_training_time"] = f3_png

    # -------------------------------------------------------------
    # Figure 4: SHAP vs. LIME Runtime Comparison
    # -------------------------------------------------------------
    shap_data = [_extract_metric(all_results[c].get("explainability_metrics", {}).get("shap", {}), "generation_time_sec", 0.0) for c in configs]
    lime_data = [_extract_metric(all_results[c].get("explainability_metrics", {}).get("lime", {}), "generation_time_sec", 0.0) for c in configs]
    shap_times = [x[0] for x in shap_data]
    shap_stds = [x[1] for x in shap_data]
    lime_times = [x[0] for x in lime_data]
    lime_stds = [x[1] for x in lime_data]

    has_shap_err = any(s > 0 for s in shap_stds)
    has_lime_err = any(s > 0 for s in lime_stds)

    x_idx = np.arange(len(configs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=300, facecolor="white")
    _apply_publication_theme(ax, grid_axis="y")

    b_shap = ax.bar(
        x_idx - width/2,
        shap_times,
        width,
        yerr=shap_stds if has_shap_err else None,
        capsize=3.5 if has_shap_err else 0,
        error_kw=dict(ecolor="#111111", lw=1.1),
        label="SHAP Generation Time",
        color="#2a9d8f",
        edgecolor="#222222",
        linewidth=1.0,
        zorder=3,
    )
    b_lime = ax.bar(
        x_idx + width/2,
        lime_times,
        width,
        yerr=lime_stds if has_lime_err else None,
        capsize=3.5 if has_lime_err else 0,
        error_kw=dict(ecolor="#111111", lw=1.1),
        label="LIME Generation Time",
        color="#7209b7",
        edgecolor="#222222",
        linewidth=1.0,
        zorder=3,
    )

    ax.set_title("Explainability Generation Runtime: SHAP vs. LIME", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(configs, fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Experimental Configuration", fontsize=12.5, fontweight="bold", labelpad=9)
    max_xai = max(shap_times + lime_times + [0.1])
    ax.set_ylim(0.0, max_xai * 1.28)

    for i, bar in enumerate(b_shap):
        h = bar.get_height()
        err_offset = shap_stds[i] if has_shap_err else 0.0
        ax.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h + err_offset), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1b4965")
    for i, bar in enumerate(b_lime):
        h = bar.get_height()
        err_offset = lime_stds[i] if has_lime_err else 0.0
        ax.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h + err_offset), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#4a0e4e")

    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="upper right")
    plt.tight_layout()
    f4_png = os.path.join(output_dir, "figure4_shap_lime_runtime.png")
    f4_pdf = os.path.join(output_dir, "figure4_shap_lime_runtime.pdf")
    plt.savefig(f4_png, bbox_inches="tight")
    plt.savefig(f4_pdf, bbox_inches="tight")
    plt.close()
    generated_plots["figure4_xai_runtime"] = f4_png

    # -------------------------------------------------------------
    # Figure 5: Explanation Fidelity and Stability (2-Panel)
    # -------------------------------------------------------------
    fid_data = [_extract_metric(all_results[c].get("explainability_metrics", {}).get("fidelity", {}), "shap_mean_fidelity_delta", 0.0) for c in configs]
    stab_data = [_extract_metric(all_results[c].get("explainability_metrics", {}).get("stability", {}), "stability_cosine_sim", 1.0) for c in configs]
    fidelity_deltas = [x[0] for x in fid_data]
    fidelity_stds = [x[1] for x in fid_data]
    stability_sims = [x[0] for x in stab_data]
    stability_stds = [x[1] for x in stab_data]

    has_fid_err = any(s > 0 for s in fidelity_stds)
    has_stab_err = any(s > 0 for s in stability_stds)

    fig, (ax_fid, ax_stab) = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=300, facecolor="white")

    # Subplot 5A: Fidelity Delta
    _apply_publication_theme(ax_fid, grid_axis="y")
    bars_f = ax_fid.bar(
        configs,
        fidelity_deltas,
        yerr=fidelity_stds if has_fid_err else None,
        capsize=4 if has_fid_err else 0,
        error_kw=dict(ecolor="#222222", lw=1.2),
        color="#e07a5f",
        width=0.52,
        edgecolor="#222222",
        linewidth=1.1,
        zorder=3,
    )
    ax_fid.set_title("(a) Top-3 Masking Sensitivity Δ", fontsize=12.5, fontweight="bold", pad=13)
    ax_fid.set_ylabel("Top-3 Masking Sensitivity Δ", fontsize=11.5, fontweight="bold")
    ax_fid.set_xlabel("Configuration", fontsize=11.5, fontweight="bold", labelpad=8)
    max_fid = max([fidelity_deltas[i] + (fidelity_stds[i] if has_fid_err else 0.0) for i in range(len(fidelity_deltas))] + [0.5])
    ax_fid.set_ylim(0.0, max_fid * 1.18)
    for i, bar in enumerate(bars_f):
        h = bar.get_height()
        err_offset = fidelity_stds[i] if has_fid_err else 0.0
        ax_fid.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h + err_offset), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # Subplot 5B: Stability Similarity
    _apply_publication_theme(ax_stab, grid_axis="y")
    bars_s = ax_stab.bar(
        configs,
        stability_sims,
        yerr=stability_stds if has_stab_err else None,
        capsize=4 if has_stab_err else 0,
        error_kw=dict(ecolor="#222222", lw=1.2),
        color="#457b9d",
        width=0.52,
        edgecolor="#222222",
        linewidth=1.1,
        zorder=3,
    )
    ax_stab.axhline(1.0, linestyle=":", color="#888888", linewidth=1.2, label="Max Stability (1.0)")
    ax_stab.set_title("(b) Explanation Stability (Cosine Similarity)", fontsize=12.5, fontweight="bold", pad=13)
    ax_stab.set_ylabel("Stability Cosine Similarity", fontsize=11.5, fontweight="bold")
    ax_stab.set_xlabel("Configuration", fontsize=11.5, fontweight="bold", labelpad=8)
    ax_stab.set_ylim(0.0, 1.25)
    for i, bar in enumerate(bars_s):
        h = bar.get_height()
        err_offset = stability_stds[i] if has_stab_err else 0.0
        ax_stab.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h + err_offset), xytext=(0, 5),
                         textcoords="offset points", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    fig.suptitle("Explainability Quality: Top-3 Masking Sensitivity and Repeated-Run Stability", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    f5_png = os.path.join(output_dir, "figure5_masking_sensitivity_stability.png")
    f5_pdf = os.path.join(output_dir, "figure5_masking_sensitivity_stability.pdf")
    plt.savefig(f5_png, bbox_inches="tight")
    plt.savefig(f5_pdf, bbox_inches="tight")
    # Also save with legacy filename for backwards compatibility
    plt.savefig(os.path.join(output_dir, "figure5_fidelity_stability.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "figure5_fidelity_stability.pdf"), bbox_inches="tight")
    plt.close()
    generated_plots["figure5_fidelity_stability"] = f5_png

    # -------------------------------------------------------------
    # Figure 6: Privacy–Utility–Explainability Trade-Off (Unclipped)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 5.8), dpi=300, facecolor="white")
    _apply_publication_theme(ax, grid_axis="")
    ax.grid(True, linestyle="--", alpha=0.45, color="#c5c5c5", zorder=0)

    xs = []
    ys = []
    for c in configs:
        x_val = all_results[c].get("performance_metrics", {}).get("total_training_time_sec", 1.0)
        x = float(x_val.get("mean", x_val)) if isinstance(x_val, dict) else float(x_val)

        y_val = all_results[c].get("recommendation_metrics", {}).get("ndcg@10", 0.0)
        y = float(y_val.get("mean", y_val)) if isinstance(y_val, dict) else float(y_val)

        fid_val = all_results[c].get("explainability_metrics", {}).get("fidelity", {}).get("shap_mean_fidelity_delta", 0.1)
        fid = float(fid_val.get("mean", fid_val)) if isinstance(fid_val, dict) else float(fid_val)

        xs.append(x)
        ys.append(y)
        bubble_size = max(200, int(fid * 280))
        color = _get_config_color(c)

        ax.scatter(x, y, s=bubble_size, color=color, label=f"{c} (Fid Δ={fid:.2f})",
                   edgecolors="#222222", linewidth=1.4, alpha=0.90, zorder=4)

        # Clear text callout box with arrow
        ax.annotate(
            c,
            xy=(x, y),
            xytext=(15, 12),
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#666666", alpha=0.94, linewidth=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color="#444444", lw=1.0),
            zorder=5,
        )

    # 25% margin padding to prevent circle clipping at edges
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_pad = max(1.5, (max_x - min_x) * 0.18)
    y_pad = max(0.06, (max_y - min_y) * 0.22)
    ax.set_xlim(max(0.0, min_x - x_pad), max_x + x_pad)
    ax.set_ylim(max(0.0, min_y - y_pad), max_y + y_pad)

    ax.set_title("Privacy–Utility–Explainability Trade-Off", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Computational Training Time (seconds) [Privacy Overhead]", fontsize=12.5, fontweight="bold", labelpad=9)
    ax.set_ylabel("Recommendation Quality (NDCG@10) [Utility]", fontsize=12.5, fontweight="bold", labelpad=9)

    legend_handles = []
    for c in configs:
        fid_val = all_results[c].get("explainability_metrics", {}).get("fidelity", {}).get("shap_mean_fidelity_delta", 0.1)
        fid = float(fid_val.get("mean", fid_val)) if isinstance(fid_val, dict) else float(fid_val)
        color = _get_config_color(c)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=f"{c} (Fid Δ = {fid:.2f})",
                markerfacecolor=color,
                markeredgecolor="#222222",
                markeredgewidth=1.1,
                markersize=9.0,
            )
        )

    ax.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=10.5,
        loc="center",
        bbox_to_anchor=(0.48, 0.55),
        title="Configurations (Bubble Size ~ Fidelity Δ)",
        title_fontsize=11.0,
        labelspacing=0.55,
        handletextpad=0.6,
    )

    plt.tight_layout()
    f6_png = os.path.join(output_dir, "figure6_privacy_utility_tradeoff.png")
    f6_pdf = os.path.join(output_dir, "figure6_privacy_utility_tradeoff.pdf")
    plt.savefig(f6_png, bbox_inches="tight")
    plt.savefig(f6_pdf, bbox_inches="tight")
    plt.close()
    return generated_plots


def generate_plots_from_10seeds(
    ten_seeds_json: str = "experiments/results/10seeds_benchmark/ten_seeds_results.json",
    multiseed_summary_json: str = "experiments/results/multiseed_benchmark/ears/aggregated_summary.json",
    output_dir: str = "Paper/figures",
) -> Dict[str, str]:
    """
    Aggregates the complete 10-seed results and generates all manuscript figures
    (Figures 1, 2, 3, 4, 5, 6) in both PDF and 300 DPI PNG matching Table III.
    """
    import json
    with open(ten_seeds_json, "r", encoding="utf-8") as f:
        ten_seeds_data = json.load(f)

    with open(multiseed_summary_json, "r", encoding="utf-8") as f:
        historical_summary = json.load(f)

    all_configs = ["E0", "E1", "E2_DP1", "E2_DP2", "E2_DP3", "E3", "E4"]
    aggregated: Dict[str, Dict[str, Any]] = {}

    for cfg in all_configs:
        ndcgs = []
        recalls = []
        train_times = []

        for s_key in ten_seeds_data:
            if cfg in ten_seeds_data[s_key]:
                c_data = ten_seeds_data[s_key][cfg]
                if "ndcg@10" in c_data:
                    ndcgs.append(c_data["ndcg@10"])
                if "recall@10" in c_data:
                    recalls.append(c_data["recall@10"])
                if "train_time_sec" in c_data:
                    train_times.append(c_data["train_time_sec"])

        # Fallback to historical if empty
        if not ndcgs and cfg in historical_summary:
            ndcg_m = historical_summary[cfg]["recommendation_metrics"]["ndcg@10"]
            ndcg_s = historical_summary[cfg]["recommendation_metrics"].get("ndcg@10_std", 0.0)
            rec_m = historical_summary[cfg]["recommendation_metrics"]["recall@10"]
            rec_s = historical_summary[cfg]["recommendation_metrics"].get("recall@10_std", 0.0)
        else:
            ndcg_m = float(np.mean(ndcgs)) if ndcgs else 0.0
            ndcg_s = float(np.std(ndcgs, ddof=1)) if len(ndcgs) > 1 else 0.0
            rec_m = float(np.mean(recalls)) if recalls else 0.0
            rec_s = float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0

        # Performance metrics
        hist_perf = historical_summary.get(cfg, {}).get("performance_metrics", {})
        if train_times:
            total_time_m = float(np.mean(train_times))
            total_time_s = float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0
        else:
            raw_tt = hist_perf.get("total_training_time_sec", 25.0)
            if isinstance(raw_tt, dict):
                total_time_m = float(raw_tt.get("mean", 25.0))
                total_time_s = float(raw_tt.get("std", 1.0))
            else:
                total_time_m = float(raw_tt)
                total_time_s = float(hist_perf.get("total_training_time_sec_std", 1.0))

        raw_agg = hist_perf.get("aggregation_time_sec", hist_perf.get("server_aggregation_time_sec", 0.0))
        if isinstance(raw_agg, dict):
            agg_time_m = float(raw_agg.get("mean", 0.0))
            agg_time_s = float(raw_agg.get("std", 0.0))
        else:
            agg_time_m = float(raw_agg)
            agg_time_s = float(hist_perf.get("aggregation_time_sec_std", hist_perf.get("server_aggregation_time_sec_std", 0.0)))

        # Explainability metrics
        hist_xai = historical_summary.get(cfg, {}).get("explainability_metrics", {})

        aggregated[cfg] = {
            "recommendation_metrics": {
                "ndcg@10": {"mean": ndcg_m, "std": ndcg_s},
                "recall@10": {"mean": rec_m, "std": rec_s},
            },
            "performance_metrics": {
                "total_training_time_sec": {"mean": total_time_m, "std": total_time_s},
                "aggregation_time_sec": {"mean": agg_time_m, "std": agg_time_s},
                "server_aggregation_time_sec": {"mean": agg_time_m, "std": agg_time_s},
            },
            "explainability_metrics": hist_xai,
        }

    plots = generate_experiment_plots(aggregated, output_dir)
    print(f"Generated {len(plots)} experiment plots in {output_dir}")
    return plots


if __name__ == "__main__":
    generate_plots_from_10seeds()

