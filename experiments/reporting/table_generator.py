from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import pandas as pd


def _fmt_val(
    d: Dict[str, Any],
    key: str,
    precision: int = 4,
    is_latex: bool = False,
    suffix: str = "",
) -> str:
    """Formats a metric value, including standard deviation if present."""
    if key not in d:
        return "—"
    val = d[key]
    if val is None:
        return "—"

    std_val = d.get(f"{key}_std")
    if isinstance(val, dict) and "mean" in val:
        m = val["mean"]
        s = val.get("std", 0.0)
    elif std_val is not None:
        m = float(val)
        s = float(std_val)
    else:
        try:
            return f"{float(val):.{precision}f}{suffix}"
        except (ValueError, TypeError):
            return f"{val}{suffix}"

    if s == 0.0:
        return f"{m:.{precision}f}{suffix}"

    if is_latex:
        return f"${m:.{precision}f} \\pm {s:.{precision}f}${suffix}"
    else:
        return f"{m:.{precision}f} ± {s:.{precision}f}{suffix}"


def generate_experiment_tables(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generates Table 1 (Recommendation), Table 2 (Computational Overhead),
    and Table 3 (Explainability Results) in Markdown and LaTeX format,
    supporting both single-run and multi-seed (mean ± std) formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    for is_latex in [False, True]:
        ext = "tex" if is_latex else "md"

        # -------------------------------------------------------------
        # Table 1: Recommendation Performance
        # -------------------------------------------------------------
        rec_rows = []
        for exp_name, res in all_results.items():
            rm = res.get("recommendation_metrics", {})
            rec_rows.append({
                "Configuration": exp_name,
                "Precision@10": _fmt_val(rm, "precision@10", 4, is_latex),
                "Recall@10": _fmt_val(rm, "recall@10", 4, is_latex),
                "NDCG@10": _fmt_val(rm, "ndcg@10", 4, is_latex),
                "MAP@10": _fmt_val(rm, "map@10", 4, is_latex),
                "Hit Rate@10": _fmt_val(rm, "hit_rate@10", 4, is_latex),
            })
        df_rec = pd.DataFrame(rec_rows)
        t1_path = os.path.join(output_dir, f"table1_recommendation_performance.{ext}")
        if is_latex:
            df_rec.to_latex(t1_path, index=False, escape=False)
            generated_files["table1_tex"] = t1_path
        else:
            df_rec.to_markdown(t1_path, index=False)
            generated_files["table1_md"] = t1_path

        # -------------------------------------------------------------
        # Table 2: Computational Performance & Privacy Overhead
        # -------------------------------------------------------------
        perf_rows = []
        e1_perf = all_results.get("E1", {}).get("performance_metrics", {})
        fl_base_time = e1_perf.get("total_training_time_sec", 1.0)
        if isinstance(fl_base_time, dict):
            fl_base_time = fl_base_time.get("mean", 1.0)
        if fl_base_time <= 0:
            fl_base_time = 1.0

        for exp_name, res in all_results.items():
            pm = res.get("performance_metrics", {})
            t_total_val = pm.get("total_training_time_sec", 0.0)
            t_total_m = t_total_val.get("mean", t_total_val) if isinstance(t_total_val, dict) else t_total_val
            overhead = ((float(t_total_m) - fl_base_time) / fl_base_time) * 100.0 if "E1" in all_results else 0.0

            perf_rows.append({
                "Configuration": exp_name,
                "Train Time (s)": _fmt_val(pm, "total_training_time_sec", 2, is_latex),
                "Local Time (s)": _fmt_val(pm, "local_training_time_sec", 2, is_latex),
                "Agg Time (s)": _fmt_val(pm, "aggregation_time_sec", 3, is_latex),
                "Comm (KB)": _fmt_val(pm, "communication_kb_sent", 1, is_latex),
                "Peak RAM (MB)": _fmt_val(pm, "peak_memory_mb", 1, is_latex),
                "HE Ciphertext (KB)": _fmt_val(pm, "he_ciphertext_size_kb", 1, is_latex),
                "DP Epsilon (ε)": f"{pm.get('dp_epsilon')}" if pm.get("dp_epsilon") is not None else "—",
                "Privacy Overhead (%)": f"{overhead:+.1f}%" if exp_name != "E0" else "Ref",
            })
        df_perf = pd.DataFrame(perf_rows)
        t2_path = os.path.join(output_dir, f"table2_computational_overhead.{ext}")
        if is_latex:
            df_perf.to_latex(t2_path, index=False, escape=False)
            generated_files["table2_tex"] = t2_path
        else:
            df_perf.to_markdown(t2_path, index=False)
            generated_files["table2_md"] = t2_path

        # -------------------------------------------------------------
        # Table 3: Explainability Results (SHAP & LIME)
        # -------------------------------------------------------------
        xai_rows = []
        for exp_name, res in all_results.items():
            xm = res.get("explainability_metrics", {})
            shap_res = xm.get("shap", {})
            lime_res = xm.get("lime", {})
            sim = xm.get("attribution_similarity_to_fl_baseline", {})
            fid = xm.get("fidelity", {})
            stab = xm.get("stability", {})

            xai_rows.append({
                "Configuration": exp_name,
                "SHAP Time (s)": _fmt_val(shap_res, "generation_time_sec", 2, is_latex),
                "LIME Time (s)": _fmt_val(lime_res, "generation_time_sec", 2, is_latex),
                "Attr Cosine Sim": _fmt_val(sim, "shap_cosine_similarity", 4, is_latex),
                "Spearman Rank ρ": _fmt_val(sim, "shap_spearman_rank_corr", 4, is_latex),
                "Top-5 Agreement": _fmt_val(sim, "shap_top5_agreement", 4, is_latex),
                "Fidelity Δ": _fmt_val(fid, "shap_mean_fidelity_delta", 4, is_latex),
                "Stability Sim": _fmt_val(stab, "stability_cosine_sim", 4, is_latex),
            })
        df_xai = pd.DataFrame(xai_rows)
        t3_path = os.path.join(output_dir, f"table3_explainability_results.{ext}")
        if is_latex:
            df_xai.to_latex(t3_path, index=False, escape=False)
            generated_files["table3_tex"] = t3_path
        else:
            df_xai.to_markdown(t3_path, index=False)
            generated_files["table3_md"] = t3_path

    return generated_files


def generate_cross_dataset_table(
    ears_results: Dict[str, Dict[str, Any]],
    movielens_results: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    """
    Generates Table 6: Cross-Dataset Performance Comparison (EARS vs. MovieLens-100k).
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    common_configs = [c for c in ears_results.keys() if c in movielens_results]
    if not common_configs:
        common_configs = list(ears_results.keys())

    for is_latex in [False, True]:
        ext = "tex" if is_latex else "md"
        rows = []
        for cfg in common_configs:
            e_res = ears_results.get(cfg, {})
            m_res = movielens_results.get(cfg, {})

            e_rm = e_res.get("recommendation_metrics", {})
            m_rm = m_res.get("recommendation_metrics", {})

            e_pm = e_res.get("performance_metrics", {})
            m_pm = m_res.get("performance_metrics", {})

            e_xm = e_res.get("explainability_metrics", {})
            m_xm = m_res.get("explainability_metrics", {})

            rows.append({
                "Config": cfg,
                "EARS NDCG@10": _fmt_val(e_rm, "ndcg@10", 4, is_latex),
                "ML-100k NDCG@10": _fmt_val(m_rm, "ndcg@10", 4, is_latex),
                "EARS Recall@10": _fmt_val(e_rm, "recall@10", 4, is_latex),
                "ML-100k Recall@10": _fmt_val(m_rm, "recall@10", 4, is_latex),
                "EARS Train (s)": _fmt_val(e_pm, "total_training_time_sec", 2, is_latex),
                "ML-100k Train (s)": _fmt_val(m_pm, "total_training_time_sec", 2, is_latex),
                "EARS Fid Δ": _fmt_val(e_xm.get("fidelity", {}), "shap_mean_fidelity_delta", 3, is_latex),
                "ML-100k Fid Δ": _fmt_val(m_xm.get("fidelity", {}), "shap_mean_fidelity_delta", 3, is_latex),
            })

        df = pd.DataFrame(rows)
        t6_path = os.path.join(output_dir, f"table6_cross_dataset_comparison.{ext}")
        if is_latex:
            df.to_latex(t6_path, index=False, escape=False)
            generated_files["table6_tex"] = t6_path
        else:
            df.to_markdown(t6_path, index=False)
            generated_files["table6_md"] = t6_path

    return generated_files

