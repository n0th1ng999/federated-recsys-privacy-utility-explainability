"""
Loading and statistics for the revised results (results/v2, results/v2_timing).

Everything that appears in the manuscript is computed here from the per-seed JSON
files; nothing is typed in by hand.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

PRIMARY = "ndcg@10"


def load_runs(root: str, dataset: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """{config: {seed: result_json}} for one dataset."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for path in glob.glob(os.path.join(root, dataset, "*", "seed_*.json")):
        cfg = os.path.basename(os.path.dirname(path))
        if cfg in ("baselines", "controls", "plausibility"):
            continue
        seed = int(os.path.basename(path)[5:-5])
        with open(path, "r", encoding="utf-8") as f:
            out.setdefault(cfg, {})[seed] = json.load(f)
    return out


def load_baselines(root: str, dataset: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for path in glob.glob(os.path.join(root, dataset, "baselines", "seed_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            out[int(os.path.basename(path)[5:-5])] = json.load(f)
    return out


def load_subdir(root: str, dataset: str, sub: str) -> Dict[int, Dict[str, Any]]:
    """Per-seed JSON files under <root>/<dataset>/<sub>/seed_<n>.json (controls, plausibility)."""
    out = {}
    for path in glob.glob(os.path.join(root, dataset, sub, "seed_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            out[int(os.path.basename(path)[5:-5])] = json.load(f)
    return out


def load_reconstruction(root: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    for path in glob.glob(os.path.join(root, "attacks", "reconstruction_seed_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            out[int(os.path.basename(path).split("_")[-1][:-5])] = json.load(f)
    return out


def describe(values: List[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": float(v.mean()), "sd": sd, "ci_lo": float(v.mean() - half), "ci_hi": float(v.mean() + half)}


def paired(a: Dict[int, float], b: Dict[int, float], margin: Optional[float] = None) -> Dict[str, Any]:
    """
    Paired comparison b - a over the seeds present in both. Returns mean difference,
    SD, 95% t-interval, paired t-test, Wilcoxon signed-rank test and, if a margin is
    given, TOST (two one-sided t-tests) for equivalence within +-margin.
    The observed-SD minimum detectable effect is reported as descriptive only.
    """
    seeds = sorted(set(a) & set(b))
    d = np.asarray([b[s] - a[s] for s in seeds], dtype=float)
    n = d.size
    res: Dict[str, Any] = {"seeds": seeds, "n": int(n)}
    if n < 2:
        return res
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    res.update({
        "mean_diff": mean, "sd_diff": sd, "se": se,
        "ci95": [mean - tcrit * se, mean + tcrit * se],
        "t": mean / se if se > 0 else float("inf"),
        "p_ttest": float(stats.ttest_1samp(d, 0.0).pvalue) if se > 0 else 0.0,
        "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        "cohens_dz": mean / sd if sd > 0 else float("nan"),
        "mde80_observed_sd": float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * se),
    })
    if margin is not None and se > 0:
        t_lower = (mean + margin) / se
        t_upper = (mean - margin) / se
        p_lower = float(1 - stats.t.cdf(t_lower, n - 1))
        p_upper = float(stats.t.cdf(t_upper, n - 1))
        res["tost"] = {
            "margin": margin, "t_lower": t_lower, "t_upper": t_upper,
            "p_lower": p_lower, "p_upper": p_upper, "p_tost": max(p_lower, p_upper),
            "ci90": [mean - stats.t.ppf(0.95, n - 1) * se, mean + stats.t.ppf(0.95, n - 1) * se],
            "equivalent_at_0.05": max(p_lower, p_upper) < 0.05,
        }
    return res


def prospective_power_paired(n: int, sd_assumed: float, effect: float, alpha: float = 0.05) -> float:
    """Power of a two-sided paired t-test for a TRUE difference `effect` given an ASSUMED SD."""
    df = n - 1
    nc = effect / (sd_assumed / math.sqrt(n))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc))


def metric_by_seed(runs: Dict[int, Dict[str, Any]], getter) -> Dict[int, float]:
    out = {}
    for seed, r in runs.items():
        try:
            v = getter(r)
        except (KeyError, TypeError):
            continue
        if v is not None:
            out[seed] = float(v)
    return out


def rec(metric: str):
    return lambda r: r["recommendation_metrics"][metric]


def perf(metric: str):
    return lambda r: r["performance_metrics"][metric]
