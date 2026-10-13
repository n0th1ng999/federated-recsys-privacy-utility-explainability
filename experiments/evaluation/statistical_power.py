"""
Statistical power analysis, paired t-tests, and Two One-Sided Tests (TOST)
equivalence verification for federated recommendation benchmarks.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.stats as stats


def compute_paired_stats(
    vec_a: List[float] | np.ndarray,
    vec_b: List[float] | np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes paired descriptive statistics, paired difference, standard error,
    Student's t 95% confidence interval, and Cohen's d effect size.
    """
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Vectors must have identical length: {len(a)} vs {len(b)}")
    n = len(a)
    diff = b - a

    mean_a = float(np.mean(a))
    std_a = float(np.std(a, ddof=1))
    mean_b = float(np.mean(b))
    std_b = float(np.std(b, ddof=1))

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = std_diff / math.sqrt(n)

    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Cohen's d for paired samples (dz)
    cohens_d = mean_diff / max(1e-12, std_diff)

    # Standard paired t-test against zero difference
    t_stat = mean_diff / max(1e-12, se_diff)
    p_val_zero = 2.0 * float(1.0 - stats.t.cdf(abs(t_stat), df=df))

    return {
        "n": n,
        "df": df,
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_margin": t_crit * se_diff,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value_zero": p_val_zero,
    }


def compute_tost_equivalence(
    vec_a: List[float] | np.ndarray,
    vec_b: List[float] | np.ndarray,
    margin: float = 0.01,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Executes Two One-Sided Tests (TOST) for practical equivalence within [-margin, +margin].
    Null H01: mu_diff <= -margin
    Null H02: mu_diff >= +margin
    Rejection of both nulls at level alpha establishes equivalence.
    """
    paired = compute_paired_stats(vec_a, vec_b, alpha=alpha)
    n = paired["n"]
    df = paired["df"]
    mean_diff = paired["mean_diff"]
    se_diff = paired["se_diff"]

    # Test 1: against lower bound -margin
    t1 = (mean_diff - (-margin)) / max(1e-12, se_diff)
    p1 = float(1.0 - stats.t.cdf(t1, df=df))

    # Test 2: against upper bound +margin
    t2 = (mean_diff - margin) / max(1e-12, se_diff)
    p2 = float(stats.t.cdf(t2, df=df))

    p_tost = max(p1, p2)
    is_equivalent = p_tost < alpha

    # Post-hoc power estimation for TOST
    power = compute_tost_power(
        n=n,
        sd=paired["std_diff"],
        margin=margin,
        alpha=alpha,
        true_diff=mean_diff,
    )

    return {
        "n": n,
        "df": df,
        "margin": margin,
        "mean_diff": mean_diff,
        "std_diff": paired["std_diff"],
        "se_diff": se_diff,
        "ci_lower": paired["ci_lower"],
        "ci_upper": paired["ci_upper"],
        "t1": float(t1),
        "p1": float(p1),
        "t2": float(t2),
        "p2": float(p2),
        "p_tost": float(p_tost),
        "is_equivalent": bool(is_equivalent),
        "observed_power": float(power),
    }


def compute_tost_power(
    n: int,
    sd: float,
    margin: float = 0.01,
    alpha: float = 0.05,
    true_diff: float = 0.0,
) -> float:
    """
    Calculates statistical power of the paired TOST equivalence test.
    Uses non-central t distribution approximations.
    """
    if n < 2 or sd <= 0:
        return 0.0

    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
    se = sd / math.sqrt(n)

    # Non-centrality parameters
    ncp1 = (margin + true_diff) / se
    ncp2 = (margin - true_diff) / se

    # Normal approximation with continuity
    z_alpha = float(stats.norm.ppf(1.0 - alpha))
    arg1 = (margin - abs(true_diff)) / se - z_alpha
    arg2 = (margin + abs(true_diff)) / se - z_alpha

    power_approx = float(stats.norm.cdf(arg1) + stats.norm.cdf(arg2) - 1.0)
    return float(np.clip(power_approx, 0.0, 1.0))


def compute_minimum_detectable_effect(
    n: int,
    sd: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> float:
    """
    Computes minimum detectable equivalence margin for power = 0.80.
    """
    if n < 2 or sd <= 0:
        return float("inf")

    z_alpha = float(stats.norm.ppf(1.0 - alpha))
    z_beta = float(stats.norm.ppf(power))
    se = sd / math.sqrt(n)
    return float((z_alpha + z_beta) * se)


def fit_dp_utility_decay(
    sigmas: List[float],
    ndcg_values: List[float],
) -> Dict[str, Any]:
    """
    Fits linear regression of NDCG degradation against noise multiplier sigma.
    """
    x = np.asarray(sigmas, dtype=np.float64)
    y = np.asarray(ndcg_values, dtype=np.float64)
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "stderr": float(res.stderr),
    }
