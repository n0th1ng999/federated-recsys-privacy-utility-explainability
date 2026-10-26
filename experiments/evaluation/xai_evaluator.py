from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import lime.lime_tabular
import numpy as np
import scipy.stats
import shap
import torch

from support.feature_contract import (
    COLLAB_BASE_CHANNELS,
    COLLAB_CHANNEL_SUPPORT_FEATURES,
    COLLABORATIVE_CHANNEL_SUPPORT_FEATURES,
    PAIR_CONTEXT_FEATURE_NAMES,
    USER_STATE_INTENT_SUMMARY_FEATURES,
    USER_STATE_OWNERSHIP_SUMMARY_FEATURES,
)


# Ordered semantic feature names (Total: 8 + 6 + 6 + 23 = 43 features)
EXPLAINABLE_FEATURE_NAMES = (
    list(USER_STATE_INTENT_SUMMARY_FEATURES)
    + list(USER_STATE_OWNERSHIP_SUMMARY_FEATURES)
    + list(COLLABORATIVE_CHANNEL_SUPPORT_FEATURES)
    + list(PAIR_CONTEXT_FEATURE_NAMES)
)

INTENT_LEN = len(USER_STATE_INTENT_SUMMARY_FEATURES)
OWNERSHIP_LEN = len(USER_STATE_OWNERSHIP_SUMMARY_FEATURES)
COLLAB_LEN = len(COLLABORATIVE_CHANNEL_SUPPORT_FEATURES)
PAIR_LEN = len(PAIR_CONTEXT_FEATURE_NAMES)


class TabularRecSysModelWrapper:
    """
    Exposes CoreHybridRecSys as f(x) on the 43 tabular summary features so that
    KernelSHAP and LIME can query it.

    The sequence (cluster) inputs are not part of the explained feature space. They
    are held FIXED at the values of `template_sample` (the instance being explained),
    so for that instance f(x) equals the real model's ranking score. Zeroing them
    would explain a different function.

    Every call is counted (`num_calls`, `num_rows`) so explanation budgets are
    reported as rows actually processed by the model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        template_sample: Dict[str, Any],
    ):
        self.model = model
        self.device = device
        self.model.eval()

        def _seq(val):
            t = val if isinstance(val, torch.Tensor) else torch.as_tensor(np.asarray(val))
            t = t.float()
            return t if t.dim() == 2 else t.reshape(t.shape[-2], t.shape[-1])

        self.intent_clusters = _seq(template_sample["user_state"]["intent_clusters"]).to(device)
        self.ownership_clusters = _seq(template_sample["user_state"]["ownership_clusters"]).to(device)
        self.collab_clusters = _seq(template_sample["collaborative"]["clusters"]).to(device)
        self.num_calls = 0
        self.num_rows = 0

    def reset_counters(self) -> None:
        self.num_calls = 0
        self.num_rows = 0

    def __call__(self, x_np: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_np, dtype=np.float64)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        b_size = x_np.shape[0]
        self.num_calls += 1
        self.num_rows += b_size

        intent_x = torch.from_numpy(x_np[:, :INTENT_LEN]).float().to(self.device)
        ownership_x = torch.from_numpy(x_np[:, INTENT_LEN : INTENT_LEN + OWNERSHIP_LEN]).float().to(self.device)
        collab_x = torch.from_numpy(x_np[:, INTENT_LEN + OWNERSHIP_LEN : INTENT_LEN + OWNERSHIP_LEN + COLLAB_LEN]).float().to(self.device)
        pair_x = torch.from_numpy(x_np[:, INTENT_LEN + OWNERSHIP_LEN + COLLAB_LEN :]).float().to(self.device)

        inputs = {
            "user_state": {
                "intent_summary": intent_x,
                "ownership_summary": ownership_x,
                "intent_clusters": self.intent_clusters.unsqueeze(0).expand(b_size, -1, -1),
                "ownership_clusters": self.ownership_clusters.unsqueeze(0).expand(b_size, -1, -1),
            },
            "collaborative": {
                "channel_support": collab_x,
                "clusters": self.collab_clusters.unsqueeze(0).expand(b_size, -1, -1),
            },
            "pair_context": {
                "context": pair_x,
            },
        }

        with torch.no_grad():
            _, rank_score, _ = self.model(inputs)
        return rank_score.cpu().numpy().flatten()


def _to_flat_numpy(val: Any) -> np.ndarray:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().flatten()
    return np.asarray(val, dtype=np.float64).flatten()


def extract_tabular_features_from_sample(sample: Dict[str, Any]) -> np.ndarray:
    """Extracts a flat 43-element float numpy vector from a dataset sample dictionary."""
    intent_summary = _to_flat_numpy(sample["user_state"]["intent_summary"])
    ownership_summary = _to_flat_numpy(sample["user_state"]["ownership_summary"])
    channel_support = _to_flat_numpy(sample["collaborative"]["channel_support"])
    pair_context = _to_flat_numpy(sample["pair_context"]["context"])
    return np.concatenate([intent_summary, ownership_summary, channel_support, pair_context]).astype(np.float64)


# -------------------------------------------------------------
# Attribution Similarity, Fidelity, and Stability Metrics
# -------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 <= 1e-12 or norm2 <= 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def spearman_rank_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    if len(v1) < 2 or np.std(v1) <= 1e-12 or np.std(v2) <= 1e-12:
        return 0.0
    rho, _ = scipy.stats.spearmanr(v1, v2)
    return float(rho) if not np.isnan(rho) else 0.0


def top_k_feature_agreement(v1: np.ndarray, v2: np.ndarray, k: int = 5) -> float:
    top1 = set(np.argsort(-np.abs(v1))[:k])
    top2 = set(np.argsort(-np.abs(v2))[:k])
    intersection = len(top1 & top2)
    union = len(top1 | top2)
    return float(intersection / union) if union > 0 else 0.0


# -------------------------------------------------------------
# SHAP Evaluation
# -------------------------------------------------------------

def evaluate_shap(
    model: torch.nn.Module,
    eval_samples: List[Dict[str, Any]],
    background_samples: List[Dict[str, Any]],
    device: torch.device,
    max_eval_samples: int = 30,
) -> Dict[str, Any]:
    """
    Runs KernelSHAP explanations on evaluation instances and measures runtime,
    model evaluations, and structured explanations.
    """
    t0 = time.perf_counter()
    wrapper = TabularRecSysModelWrapper(model, device, eval_samples[0])

    bg_x = np.stack([extract_tabular_features_from_sample(s) for s in background_samples])
    eval_instances = eval_samples[:max_eval_samples]
    test_x = np.stack([extract_tabular_features_from_sample(s) for s in eval_instances])
    print(f"  [XAI - SHAP] Running KernelExplainer on {len(eval_instances)} instances (nsamples=50)...")

    explainer = shap.KernelExplainer(wrapper, bg_x)

    # Evaluate SHAP values
    shap_values = explainer.shap_values(test_x, nsamples=50, silent=True)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    gen_time = time.perf_counter() - t0
    num_evals = explainer.nsamples if hasattr(explainer, "nsamples") else 50 * len(test_x)
    print(f"  [XAI - SHAP] Completed {len(eval_instances)} explanations in {gen_time:.2f}s ({gen_time/max(1, len(test_x)):.3f}s/instance)")

    structured_explanations: List[Dict[str, Any]] = []
    for idx, sample in enumerate(eval_instances):
        user_id = int(sample.get("user_id", sample.get("client_id", idx)))
        item_id = int(sample.get("item_id", sample.get("product_id", idx)))
        attrs = shap_values[idx]
        sorted_indices = np.argsort(-np.abs(attrs))

        for rank, f_idx in enumerate(sorted_indices, 1):
            structured_explanations.append({
                "user_id": user_id,
                "item_id": item_id,
                "feature": EXPLAINABLE_FEATURE_NAMES[f_idx],
                "shap_value": float(attrs[f_idx]),
                "rank": rank,
            })

    return {
        "generation_time_sec": round(gen_time, 3),
        "avg_time_per_explanation_sec": round(gen_time / max(1, len(test_x)), 4),
        "num_model_evaluations": int(num_evals),
        "attributions": shap_values,  # numpy array [N, num_features]
        "structured_explanations": structured_explanations,
    }


# -------------------------------------------------------------
# LIME Evaluation
# -------------------------------------------------------------

def evaluate_lime(
    model: torch.nn.Module,
    eval_samples: List[Dict[str, Any]],
    background_samples: List[Dict[str, Any]],
    device: torch.device,
    max_eval_samples: int = 30,
) -> Dict[str, Any]:
    """
    Runs LIME tabular explanations on the exact same evaluation instances.
    """
    t0 = time.perf_counter()
    wrapper = TabularRecSysModelWrapper(model, device, eval_samples[0])

    bg_x = np.stack([extract_tabular_features_from_sample(s) for s in background_samples])
    eval_instances = eval_samples[:max_eval_samples]
    print(f"  [XAI - LIME] Running LimeTabularExplainer on {len(eval_instances)} instances...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        bg_x,
        feature_names=list(EXPLAINABLE_FEATURE_NAMES),
        mode="regression",
        verbose=False,
    )

    lime_weights_matrix = []
    structured_explanations: List[Dict[str, Any]] = []
    log_interval = max(5, len(eval_instances) // 3)

    for idx, sample in enumerate(eval_instances):
        user_id = int(sample.get("user_id", sample.get("client_id", idx)))
        item_id = int(sample.get("item_id", sample.get("product_id", idx)))
        x_vec = extract_tabular_features_from_sample(sample)

        exp = explainer.explain_instance(x_vec, wrapper, num_features=len(EXPLAINABLE_FEATURE_NAMES))
        # Build 1D weight vector
        weight_vec = np.zeros(len(EXPLAINABLE_FEATURE_NAMES), dtype=np.float64)
        for f_name, w in exp.as_list():
            # Match feature name substring
            for f_idx, full_name in enumerate(EXPLAINABLE_FEATURE_NAMES):
                if full_name in f_name:
                    weight_vec[f_idx] = float(w)
                    break
        lime_weights_matrix.append(weight_vec)

        sorted_indices = np.argsort(-np.abs(weight_vec))
        for rank, f_idx in enumerate(sorted_indices, 1):
            structured_explanations.append({
                "user_id": user_id,
                "item_id": item_id,
                "feature": EXPLAINABLE_FEATURE_NAMES[f_idx],
                "lime_weight": float(weight_vec[f_idx]),
                "rank": rank,
            })

        if (idx + 1) % log_interval == 0 or (idx + 1) == len(eval_instances):
            pct = ((idx + 1) / len(eval_instances)) * 100.0
            print(f"    -> LIME explained {idx+1:02d}/{len(eval_instances)} ({pct:5.1f}%) | Elapsed: {time.perf_counter() - t0:.1f}s")

    gen_time = time.perf_counter() - t0
    print(f"  [XAI - LIME] Completed {len(eval_instances)} explanations in {gen_time:.2f}s ({gen_time/max(1, len(eval_instances)):.3f}s/instance)")
    weights_np = np.stack(lime_weights_matrix)

    return {
        "generation_time_sec": round(gen_time, 3),
        "avg_time_per_explanation_sec": round(gen_time / max(1, len(eval_instances)), 4),
        "num_model_evaluations": len(eval_instances) * 500,  # default lime num_samples
        "attributions": weights_np,
        "structured_explanations": structured_explanations,
    }


# -------------------------------------------------------------
# Fidelity & Stability Evaluation
# -------------------------------------------------------------

def evaluate_explanation_fidelity(
    model: torch.nn.Module,
    eval_samples: List[Dict[str, Any]],
    attributions: np.ndarray,
    device: torch.device,
    top_k_perturb: int = 3,
) -> Dict[str, Any]:
    """
    Measures explanation fidelity:
    1. Generates original prediction
    2. Masks top-k most important features (zeros them out)
    3. Generates perturbed prediction
    4. Computes mean absolute prediction difference: |y_orig - y_pert|
    """
    wrapper = TabularRecSysModelWrapper(model, device, eval_samples[0])
    test_x = np.stack([extract_tabular_features_from_sample(s) for s in eval_samples[: len(attributions)]])

    original_preds = wrapper(test_x)
    perturbed_preds = []
    differences = []

    for idx, x_vec in enumerate(test_x):
        top_indices = np.argsort(-np.abs(attributions[idx]))[:top_k_perturb]
        perturbed_x = x_vec.copy()
        perturbed_x[top_indices] = 0.0  # Feature removal / perturbation

        pred_pert = wrapper(perturbed_x.reshape(1, -1))[0]
        perturbed_preds.append(float(pred_pert))
        differences.append(float(abs(original_preds[idx] - pred_pert)))

    return {
        "mean_fidelity_delta": float(np.mean(differences)),
        "max_fidelity_delta": float(np.max(differences)),
        "min_fidelity_delta": float(np.min(differences)),
        "predictions": [
            {
                "original_prediction": float(orig),
                "perturbed_prediction": float(pert),
                "prediction_difference": float(diff),
            }
            for orig, pert, diff in zip(original_preds, perturbed_preds, differences)
        ],
    }


def evaluate_explanation_stability(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    background_samples: List[Dict[str, Any]],
    device: torch.device,
    explainer_type: str = "shap",
    num_runs: int = 3,
) -> Dict[str, float]:
    """
    Measures explanation stability by repeating explanation generation with different seeds.
    Computes pairwise Cosine similarity and Spearman rank correlation across runs.
    """
    x_vec = extract_tabular_features_from_sample(sample).reshape(1, -1)
    bg_x = np.stack([extract_tabular_features_from_sample(s) for s in background_samples])
    wrapper = TabularRecSysModelWrapper(model, device, sample)

    run_attributions = []

    for seed in range(num_runs):
        np.random.seed(seed + 100)
        if explainer_type == "shap":
            exp = shap.KernelExplainer(wrapper, bg_x)
            attrs = exp.shap_values(x_vec, nsamples=40, silent=True)
            if isinstance(attrs, list):
                attrs = attrs[0]
            run_attributions.append(attrs.flatten())
        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                bg_x,
                feature_names=list(EXPLAINABLE_FEATURE_NAMES),
                mode="regression",
                random_state=seed + 100,
            )
            exp = explainer.explain_instance(x_vec.flatten(), wrapper, num_features=len(EXPLAINABLE_FEATURE_NAMES))
            vec = np.zeros(len(EXPLAINABLE_FEATURE_NAMES))
            for f_name, w in exp.as_list():
                for f_idx, full_name in enumerate(EXPLAINABLE_FEATURE_NAMES):
                    if full_name in f_name:
                        vec[f_idx] = float(w)
                        break
            run_attributions.append(vec)

    # Compute pairwise similarities
    cos_sims = []
    spearman_corrs = []
    agreements = []

    for i in range(len(run_attributions)):
        for j in range(i + 1, len(run_attributions)):
            cos_sims.append(cosine_similarity(run_attributions[i], run_attributions[j]))
            spearman_corrs.append(spearman_rank_correlation(run_attributions[i], run_attributions[j]))
            agreements.append(top_k_feature_agreement(run_attributions[i], run_attributions[j], k=5))

    return {
        "stability_cosine_sim": float(np.mean(cos_sims)) if cos_sims else 1.0,
        "stability_spearman_corr": float(np.mean(spearman_corrs)) if spearman_corrs else 1.0,
        "stability_top5_agreement": float(np.mean(agreements)) if agreements else 1.0,
    }


def evaluate_permutation_fidelity(
    model: torch.nn.Module,
    eval_samples: List[Dict[str, Any]],
    background_samples: List[Dict[str, Any]],
    attributions: np.ndarray,
    device: torch.device,
    top_k_perturb: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Measures explanation fidelity via marginal permutation masking:
    Instead of zero-masking, the top-k most important features are replaced
    with values drawn from their empirical marginal distribution in background_samples.
    """
    np.random.seed(seed)
    wrapper = TabularRecSysModelWrapper(model, device, eval_samples[0])
    test_x = np.stack([extract_tabular_features_from_sample(s) for s in eval_samples[: len(attributions)]])
    bg_x = np.stack([extract_tabular_features_from_sample(s) for s in background_samples])

    original_preds = wrapper(test_x)
    perturbed_preds = []
    differences = []

    for idx, x_vec in enumerate(test_x):
        top_indices = np.argsort(-np.abs(attributions[idx]))[:top_k_perturb]
        perturbed_x = x_vec.copy()
        perm_donor_idx = np.random.randint(0, len(bg_x))
        perturbed_x[top_indices] = bg_x[perm_donor_idx, top_indices]

        pred_pert = wrapper(perturbed_x.reshape(1, -1))[0]
        perturbed_preds.append(float(pred_pert))
        differences.append(float(abs(original_preds[idx] - pred_pert)))

    return {
        "mean_permutation_delta": float(np.mean(differences)),
        "max_permutation_delta": float(np.max(differences)),
        "min_permutation_delta": float(np.min(differences)),
    }


def evaluate_deletion_auc(
    model: torch.nn.Module,
    eval_samples: List[Dict[str, Any]],
    attributions: np.ndarray,
    device: torch.device,
    max_k: int = 10,
) -> Dict[str, Any]:
    """
    Measures explanation faithfulness via Deletion AUC:
    Features are progressively masked in descending order of attribution magnitude (k=1..max_k).
    Lower Deletion AUC indicates that the most important features cause faster prediction decay.
    """
    wrapper = TabularRecSysModelWrapper(model, device, eval_samples[0])
    test_x = np.stack([extract_tabular_features_from_sample(s) for s in eval_samples[: len(attributions)]])

    original_preds = wrapper(test_x)
    instance_curves = []

    for idx, x_vec in enumerate(test_x):
        sorted_indices = np.argsort(-np.abs(attributions[idx]))
        curve = [1.0]
        orig_score = max(1e-8, abs(float(original_preds[idx])))

        for k in range(1, min(max_k + 1, len(sorted_indices))):
            perturbed_x = x_vec.copy()
            perturbed_x[sorted_indices[:k]] = 0.0
            pred_k = wrapper(perturbed_x.reshape(1, -1))[0]
            rel_score = float(abs(pred_k) / orig_score)
            curve.append(rel_score)

        auc_val = float(np.trapz(curve, dx=1.0 / max_k))
        instance_curves.append(auc_val)

    return {
        "mean_deletion_auc": float(np.mean(instance_curves)),
        "std_deletion_auc": float(np.std(instance_curves)),
    }


def evaluate_inter_method_agreement(
    shap_attributions: np.ndarray,
    lime_attributions: np.ndarray,
    k: int = 5,
) -> Dict[str, float]:
    """
    Computes agreement between KernelSHAP and LIME on the identical evaluation instances:
    1. Spearman rank correlation rho(SHAP, LIME)
    2. Cosine similarity
    3. Top-k feature set agreement (Jaccard)
    """
    n_samples = min(len(shap_attributions), len(lime_attributions))
    spearman_corrs = []
    cos_sims = []
    topk_agrs = []

    for i in range(n_samples):
        s_vec = shap_attributions[i]
        l_vec = lime_attributions[i]
        spearman_corrs.append(spearman_rank_correlation(s_vec, l_vec))
        cos_sims.append(cosine_similarity(s_vec, l_vec))
        topk_agrs.append(top_k_feature_agreement(s_vec, l_vec, k=k))

    return {
        "inter_method_spearman_rho": float(np.mean(spearman_corrs)),
        "inter_method_cosine_sim": float(np.mean(cos_sims)),
        "inter_method_topk_jaccard": float(np.mean(topk_agrs)),
    }

