from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


def compute_ranking_metrics_for_user(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: List[int] = [5, 10, 20],
    relevance_threshold: float = 2.5,
) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, NDCG@K, MAP@K, and Hit Rate@K for a single user.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.size < 2:
        return {}

    order = np.argsort(-y_pred)
    sorted_true = y_true[order]

    # Binary relevance indicator for metrics that require binary hits
    binary_rel = (sorted_true >= relevance_threshold).astype(np.float32)
    total_relevant = float(np.sum(binary_rel))

    metrics: Dict[str, float] = {}

    for k in k_values:
        kk = min(k, len(sorted_true))
        top_k_binary = binary_rel[:kk]
        hits = float(np.sum(top_k_binary))

        # 1. Hit Rate@K
        metrics[f"hit_rate@{k}"] = 1.0 if hits > 0 else 0.0

        # 2. Precision@K
        metrics[f"precision@{k}"] = float(hits / k)

        # 3. Recall@K
        if total_relevant > 0:
            metrics[f"recall@{k}"] = float(hits / total_relevant)
        else:
            metrics[f"recall@{k}"] = float("nan")

        # 4. NDCG@K
        # Uses graded relevance (normalized to non-negative)
        graded_rel = np.clip(sorted_true[:kk], 0.0, None)
        ideal_graded = np.sort(np.clip(y_true, 0.0, None))[::-1][:kk]

        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float(np.sum((np.power(2.0, graded_rel) - 1.0) * discounts))
        idcg = float(np.sum((np.power(2.0, ideal_graded) - 1.0) * discounts))

        if idcg > 1e-12:
            metrics[f"ndcg@{k}"] = float(dcg / idcg)
        else:
            metrics[f"ndcg@{k}"] = float("nan")

        # 5. MAP@K (Average Precision @ K)
        if total_relevant > 0:
            cum_hits = np.cumsum(top_k_binary)
            precisions = cum_hits / np.arange(1, kk + 1)
            ap = float(np.sum(precisions * top_k_binary) / min(k, total_relevant))
            metrics[f"map@{k}"] = ap
        else:
            metrics[f"map@{k}"] = float("nan")

    return metrics


def evaluate_model_ranking(
    model: torch.nn.Module,
    eval_user_samples: Dict[int, List[int]],
    dataset: Any,
    device: torch.device,
    k_values: List[int] = [5, 10, 20],
    primary_k: int = 10,
    relevance_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates the model across all evaluation users and averages the ranking metrics.
    With the temporal holdout, labels are binary (1 = item purchased after the cutoff,
    0 = sampled never-interacted item), so the default threshold is 0.5.
    """
    model.eval()
    user_metrics_list: List[Dict[str, float]] = []
    total_users = len(eval_user_samples)
    log_interval = max(5, total_users // 5)

    with torch.no_grad():
        for u_idx, (user_id, sample_indices) in enumerate(eval_user_samples.items()):
            if len(sample_indices) < 2:
                continue

            # Collate sample batch for this user
            samples = [dataset[idx] for idx in sample_indices]
            from experiments.data.partitioner import custom_collate
            batch = custom_collate(samples)

            # Move branch inputs to device
            inputs = {
                "user_state": {k: v.to(device) for k, v in batch["user_state"].items()},
                "collaborative": {k: v.to(device) for k, v in batch["collaborative"].items()},
                "pair_context": {k: v.to(device) for k, v in batch["pair_context"].items()},
            }

            purchase_logit, rank_score, conf_logit = model(inputs)
            preds = rank_score.cpu().numpy().flatten()
            labels_rank = batch["label_rank"].cpu().numpy().flatten()

            user_res = compute_ranking_metrics_for_user(
                y_true=labels_rank,
                y_pred=preds,
                k_values=k_values,
                relevance_threshold=relevance_threshold,
            )
            if user_res:
                user_metrics_list.append(user_res)

            if (u_idx + 1) % log_interval == 0 or (u_idx + 1) == total_users:
                curr_ndcg = np.mean([m[f"ndcg@{primary_k}"] for m in user_metrics_list])
                pct = ((u_idx + 1) / total_users) * 100.0
                print(f"  [Ranking Eval] User {u_idx+1:02d}/{total_users} ({pct:5.1f}%) | Current NDCG@{primary_k}: {curr_ndcg:.4f}")

    if not user_metrics_list:
        return {
            f"precision@{primary_k}": 0.0,
            f"recall@{primary_k}": 0.0,
            f"ndcg@{primary_k}": 0.0,
            f"map@{primary_k}": 0.0,
            f"hit_rate@{primary_k}": 0.0,
        }

    # Aggregate by nanmean
    aggregated: Dict[str, float] = {}
    all_keys = user_metrics_list[0].keys()
    for key in all_keys:
        vals = [um[key] for um in user_metrics_list if not np.isnan(um[key])]
        aggregated[key] = float(np.mean(vals)) if vals else 0.0
    aggregated["num_users"] = len(user_metrics_list)
    aggregated["per_user"] = {
        f"ndcg@{primary_k}": [float(um[f"ndcg@{primary_k}"]) for um in user_metrics_list],
        f"recall@{primary_k}": [float(um[f"recall@{primary_k}"]) for um in user_metrics_list],
    }

    return aggregated


def evaluate_score_baseline(
    eval_user_samples: Dict[int, List[int]],
    dataset: Any,
    item_scores: Dict[int, float],
    k_values: List[int] = [5, 10, 20],
    primary_k: int = 10,
    relevance_threshold: float = 0.5,
    random_repeats: int = 0,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Non-learned reference rankings on the same candidates: items ordered by
    `item_scores` (e.g. pre-cutoff popularity), or uniformly at random averaged over
    `random_repeats` permutations when random_repeats > 0.
    """
    rng = np.random.default_rng(seed)
    per_user = []
    for _, idxs in eval_user_samples.items():
        if len(idxs) < 2:
            continue
        y = np.asarray([float(dataset.sample_label_rank[i]) for i in idxs])
        pids = [int(dataset.sample_product_ids[i]) for i in idxs]
        if random_repeats > 0:
            runs = [compute_ranking_metrics_for_user(y, rng.random(len(y)), k_values, relevance_threshold)
                    for _ in range(random_repeats)]
            per_user.append({k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]})
        else:
            scores = np.asarray([item_scores.get(p, 0.0) for p in pids]) + 1e-9 * rng.random(len(pids))
            per_user.append(compute_ranking_metrics_for_user(y, scores, k_values, relevance_threshold))
    out = {}
    for key in per_user[0]:
        vals = [u[key] for u in per_user if not np.isnan(u[key])]
        out[key] = float(np.mean(vals)) if vals else 0.0
    out["num_users"] = len(per_user)
    return out
