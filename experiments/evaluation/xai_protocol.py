"""
Explanation protocol used in the revised manuscript.

Separates three questions that the first version conflated:
  1. Cost: how much model work each explainer performs (coalitions vs. model calls
     vs. rows actually processed), and latency at matched row budgets.
  2. Fidelity to the PRIVATE model itself: removal curves on the model being
     explained, under SHAP order, reverse order and random order, with two
     replacement schemes (marginal and approximately conditional) and a check of
     how far the intervened inputs fall from the data distribution.
  3. Agreement with the centralised model's explanations (E0): a divergence
     measure, not a fidelity measure.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import lime.lime_tabular
import numpy as np
import shap

from experiments.evaluation.xai_evaluator import (
    EXPLAINABLE_FEATURE_NAMES,
    TabularRecSysModelWrapper,
    cosine_similarity,
    extract_tabular_features_from_sample,
    spearman_rank_correlation,
    top_k_feature_agreement,
)

NUM_FEATURES = len(EXPLAINABLE_FEATURE_NAMES)


def _bootstrap_ci(values: Sequence[float], seed: int, n_boot: int = 2000, alpha: float = 0.05) -> List[float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, vals.size, size=(n_boot, vals.size))].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def _summary(values: Sequence[float], seed: int) -> Dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
            "ci95": _bootstrap_ci(vals, seed), "n": int(vals.size)}


# ----------------------------------------------------------------------------
# Explainers with measured budgets
# ----------------------------------------------------------------------------

def explain_kernel_shap(model, device, instances, background_x: np.ndarray, nsamples: int, seed: int) -> Dict[str, Any]:
    """
    KernelSHAP with `nsamples` coalitions per instance. Each coalition is evaluated
    against every background row, so rows processed ~= nsamples * len(background).
    """
    attributions, rows, calls, times = [], [], [], []
    for i, sample in enumerate(instances):
        np.random.seed(seed + i)
        wrapper = TabularRecSysModelWrapper(model, device, sample)
        x = extract_tabular_features_from_sample(sample).reshape(1, -1)
        explainer = shap.KernelExplainer(wrapper, background_x)
        wrapper.reset_counters()  # exclude the background expectation call made in __init__
        t0 = time.perf_counter()
        vals = explainer.shap_values(x, nsamples=nsamples, silent=True)
        times.append(time.perf_counter() - t0)
        vals = vals[0] if isinstance(vals, list) else vals
        attributions.append(np.asarray(vals).reshape(-1))
        rows.append(wrapper.num_rows)
        calls.append(wrapper.num_calls)
    return {
        "attributions": np.stack(attributions),
        "budget": {
            "coalitions_per_instance": int(nsamples),
            "background_rows": int(len(background_x)),
            "predict_calls_per_instance": float(np.mean(calls)),
            "rows_per_instance": float(np.mean(rows)),
        },
        "seconds_per_instance": float(np.mean(times)),
        "microseconds_per_row": float(1e6 * np.sum(times) / max(1, np.sum(rows))),
    }


def explain_lime(model, device, instances, training_x: np.ndarray, num_samples: int, seed: int) -> Dict[str, Any]:
    """LIME (tabular, regression) with an explicit `num_samples` perturbations per instance."""
    attributions, rows, calls, times = [], [], [], []
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_x,
        feature_names=list(EXPLAINABLE_FEATURE_NAMES),
        mode="regression",
        discretize_continuous=False,
        random_state=seed,
        verbose=False,
    )
    for sample in instances:
        wrapper = TabularRecSysModelWrapper(model, device, sample)
        x = extract_tabular_features_from_sample(sample)
        t0 = time.perf_counter()
        exp = explainer.explain_instance(x, wrapper, num_features=NUM_FEATURES, num_samples=num_samples)
        times.append(time.perf_counter() - t0)
        vec = np.zeros(NUM_FEATURES)
        for f_idx, w in next(iter(exp.local_exp.values())):
            vec[int(f_idx)] = float(w)
        attributions.append(vec)
        rows.append(wrapper.num_rows)
        calls.append(wrapper.num_calls)
    return {
        "attributions": np.stack(attributions),
        "budget": {
            "perturbations_per_instance": int(num_samples),
            "predict_calls_per_instance": float(np.mean(calls)),
            "rows_per_instance": float(np.mean(rows)),
        },
        "seconds_per_instance": float(np.mean(times)),
        "microseconds_per_row": float(1e6 * np.sum(times) / max(1, np.sum(rows))),
    }


# ----------------------------------------------------------------------------
# Interventions
# ----------------------------------------------------------------------------

class Replacer:
    """
    Replaces a subset S of features of x with reference values.

    * "marginal": each feature in S drawn independently from its empirical marginal
      in the reference pool (breaks dependence between features, can create
      off-manifold inputs).
    * "conditional": values for S copied jointly from one of the k nearest reference
      rows, with distance measured on the RETAINED features (an approximation of
      sampling from p(x_S | x_rest)).
    """

    def __init__(self, reference_x: np.ndarray, k_neighbors: int = 10):
        self.ref = reference_x
        self.mu = reference_x.mean(axis=0)
        self.sd = reference_x.std(axis=0) + 1e-8
        self.ref_z = (reference_x - self.mu) / self.sd
        self.k = k_neighbors

    def apply(self, x: np.ndarray, subset: np.ndarray, scheme: str, rng: np.random.Generator) -> np.ndarray:
        out = x.copy()
        if subset.size == 0:
            return out
        if scheme == "marginal":
            donors = rng.integers(0, len(self.ref), size=subset.size)
            out[subset] = self.ref[donors, subset]
            return out
        if scheme == "conditional":
            keep = np.setdiff1d(np.arange(x.size), subset)
            if keep.size == 0:
                donor = rng.integers(0, len(self.ref))
            else:
                xz = (x[keep] - self.mu[keep]) / self.sd[keep]
                d = np.linalg.norm(self.ref_z[:, keep] - xz, axis=1)
                nn = np.argsort(d)[: self.k]
                donor = rng.choice(nn)
            out[subset] = self.ref[donor, subset]
            return out
        raise ValueError(scheme)

    def nn_distance(self, rows: np.ndarray) -> np.ndarray:
        """Standardised distance of each row to its nearest reference row."""
        rz = (rows - self.mu) / self.sd
        d = np.linalg.norm(rz[:, None, :] - self.ref_z[None, :, :], axis=2)
        return d.min(axis=1)


def removal_curves(
    model, device, instances, attributions: np.ndarray, replacer: Replacer, output_scale: float,
    seed: int, draws: int = 10, random_orders: int = 20, max_k: int = NUM_FEATURES,
) -> Dict[str, Any]:
    """
    For each instance and each ordering (attribution, reverse, random), remove the
    first k features (k = 0..max_k) by replacement and record the effect
    e(k) = |f(x) - E[f(x with first k replaced)]| / output_scale,
    where the expectation is over `draws` replacement draws and output_scale is the
    SD of f over the reference pool. Curves are summarised by their mean over k (AUC).
    A faithful attribution yields a larger AUC than random order, and reverse order
    the smallest.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    plausibility: Dict[str, List[float]] = {}
    for scheme in ("marginal", "conditional"):
        curves = {"attribution": [], "reverse": [], "random": []}
        nn_d = []
        for idx, sample in enumerate(instances):
            wrapper = TabularRecSysModelWrapper(model, device, sample)
            x = extract_tabular_features_from_sample(sample)
            fx = float(wrapper(x)[0])
            order_attr = np.argsort(-np.abs(attributions[idx]))
            orders = {"attribution": [order_attr], "reverse": [order_attr[::-1]],
                      "random": [rng.permutation(NUM_FEATURES) for _ in range(random_orders)]}
            for name, order_list in orders.items():
                inst_curves = []
                for order in order_list:
                    rows = []
                    for k in range(max_k + 1):
                        for _ in range(draws if k > 0 else 1):
                            rows.append(replacer.apply(x, order[:k], scheme, rng))
                    preds = wrapper(np.stack(rows))
                    curve, pos = [0.0], 1
                    for k in range(1, max_k + 1):
                        curve.append(abs(fx - float(np.mean(preds[pos:pos + draws]))) / output_scale)
                        pos += draws
                    inst_curves.append(curve)
                    if name == "attribution":
                        # plausibility of intervened inputs at k = 3 and k = 10
                        nn_d.extend(replacer.nn_distance(np.stack(rows[1 + 2 * draws:1 + 3 * draws])).tolist())
                        nn_d.extend(replacer.nn_distance(np.stack(rows[1 + 9 * draws:1 + 10 * draws])).tolist())
                curves[name].append(np.mean(np.asarray(inst_curves), axis=0))
        res = {}
        for name, cs in curves.items():
            arr = np.asarray(cs)
            res[name] = {
                "mean_curve": arr.mean(axis=0).round(5).tolist(),
                "auc": _summary(arr.mean(axis=1), seed),
            }
        gap = np.asarray(curves["attribution"]).mean(axis=1) - np.asarray(curves["random"]).mean(axis=1)
        res["attribution_minus_random_auc"] = _summary(gap, seed)
        out[scheme] = res
        plausibility[scheme] = nn_d
    return {"curves": out, "intervened_nn_distance": {k: _summary(v, seed) for k, v in plausibility.items()}}


# ----------------------------------------------------------------------------
# Full protocol for one trained model
# ----------------------------------------------------------------------------

def run_xai_protocol(
    model, device, instances: List[Dict[str, Any]], background_samples: List[Dict[str, Any]],
    reference_samples: List[Dict[str, Any]], heldout_reference: List[Dict[str, Any]], seed: int,
    shap_nsamples: int = 200, lime_matched_samples: int = None, lime_default_samples: int = 5000,
    stability_instances: int = 5, stability_runs: int = 3,
) -> Dict[str, Any]:
    bg_x = np.stack([extract_tabular_features_from_sample(s) for s in background_samples])
    ref_x = np.stack([extract_tabular_features_from_sample(s) for s in reference_samples])
    held_x = np.stack([extract_tabular_features_from_sample(s) for s in heldout_reference])

    shap_res = explain_kernel_shap(model, device, instances, bg_x, shap_nsamples, seed)
    matched = int(round(shap_res["budget"]["rows_per_instance"])) if lime_matched_samples is None else lime_matched_samples
    lime_matched = explain_lime(model, device, instances, ref_x, matched, seed)
    lime_default = explain_lime(model, device, instances, ref_x, lime_default_samples, seed)

    # Stability: repeated SHAP runs with different seeds on the first instances
    stab_cos, stab_rho, stab_top5 = [], [], []
    for i, sample in enumerate(instances[:stability_instances]):
        runs = [explain_kernel_shap(model, device, [sample], bg_x, shap_nsamples, seed + 1000 * (r + 1))["attributions"][0]
                for r in range(stability_runs)]
        for a in range(stability_runs):
            for b in range(a + 1, stability_runs):
                stab_cos.append(cosine_similarity(runs[a], runs[b]))
                stab_rho.append(spearman_rank_correlation(runs[a], runs[b]))
                stab_top5.append(top_k_feature_agreement(runs[a], runs[b], k=5))

    # Output scale of the model on held-out reference inputs (for normalised effects)
    scale_wrapper = TabularRecSysModelWrapper(model, device, instances[0])
    output_scale = float(np.std(scale_wrapper(ref_x))) or 1.0

    replacer = Replacer(ref_x)
    removal = removal_curves(model, device, instances, shap_res["attributions"], replacer, output_scale, seed)
    removal_lime = removal_curves(model, device, instances, lime_matched["attributions"], replacer, output_scale, seed + 7,
                                  random_orders=5)
    # Reference: NN distance of genuine held-out rows to the reference pool
    real_nn = replacer.nn_distance(held_x)

    shap_lime_rho = [spearman_rank_correlation(a, b) for a, b in zip(shap_res["attributions"], lime_matched["attributions"])]

    return {
        "num_instances": len(instances),
        "shap": {k: v for k, v in shap_res.items() if k != "attributions"},
        "lime_matched_rows": {k: v for k, v in lime_matched.items() if k != "attributions"},
        "lime_default": {k: v for k, v in lime_default.items() if k != "attributions"},
        "stability_shap": {"cosine": _summary(stab_cos, seed), "spearman": _summary(stab_rho, seed),
                           "top5_jaccard": _summary(stab_top5, seed)},
        "fidelity_shap": removal,
        "fidelity_lime_matched": removal_lime,
        "heldout_nn_distance": _summary(real_nn, seed),
        "output_scale_sd": output_scale,
        "shap_vs_lime_spearman": _summary(shap_lime_rho, seed),
        "_shap_attributions": shap_res["attributions"],
    }


def agreement_with_reference(attr: np.ndarray, ref_attr: np.ndarray, seed: int) -> Dict[str, Any]:
    """Agreement between this model's SHAP attributions and the centralised model's (E0)."""
    cos = [cosine_similarity(a, b) for a, b in zip(attr, ref_attr)]
    rho = [spearman_rank_correlation(a, b) for a, b in zip(attr, ref_attr)]
    top5 = [top_k_feature_agreement(a, b, k=5) for a, b in zip(attr, ref_attr)]
    return {"cosine": _summary(cos, seed), "spearman": _summary(rho, seed), "top5_jaccard": _summary(top5, seed)}
