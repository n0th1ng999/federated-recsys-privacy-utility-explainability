"""
Membership-inference attacks aligned with the declared privacy unit.

Adversary (both attacks): black-box query access to the RELEASED final model
(external party or any participant), knowledge of the training loss and of the
candidate records with their labels. No shadow models: the attack score is the
negative per-record loss (Yeom et al., 2018), evaluated threshold-free (ROC AUC)
and at fixed false-positive rates.

(a) Client participation (the DP unit: one client = one user).
    Members: training clients that took part in >= 1 round.
    Non-members: users drawn from the same pool and processed identically, who
    never trained. User score = mean negative loss over the user's records.
(b) Record membership (finer than the DP unit; reported for completeness).
    Members: training records of participating clients.
    Non-members: records of the SAME clients withheld from training.

AUC < 0.5 means members have HIGHER loss than non-members. That is not leakage
in the attacker's favour by itself; an adversary who knew the direction could
flip the score. We therefore report the raw AUC with a bootstrap CI and a
permutation p-value for AUC != 0.5, plus the member/non-member loss
distributions, and do not interpret a sub-0.5 AUC as protection.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from experiments.data.partitioner import custom_collate

ATTACK_DESCRIPTION = {
    "adversary": "black-box access to the released final model; knows loss function and candidate records with labels",
    "score": "negative binary cross-entropy of the purchase head on the record label",
    "shadow_models": False,
    "threshold": "threshold-free ROC AUC; TPR reported at FPR 0.01 and 0.1",
    "uncertainty": "bootstrap 95% CI (2000 resamples of members and non-members); two-sided permutation test (2000 permutations)",
}


def per_record_losses(model, dataset, indices: List[int], device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = custom_collate([dataset[i] for i in indices[start:start + batch_size]])
            inputs = {
                "user_state": {k: v.float().to(device) for k, v in batch["user_state"].items()},
                "collaborative": {k: v.float().to(device) for k, v in batch["collaborative"].items()},
                "pair_context": {k: v.float().to(device) for k, v in batch["pair_context"].items()},
            }
            purchase_logit, _, _ = model(inputs)
            loss = F.binary_cross_entropy_with_logits(
                purchase_logit.flatten(), batch["label_reg"].float().to(device).flatten(), reduction="none"
            )
            out.append(loss.cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def _tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def score_attack(member_scores: np.ndarray, nonmember_scores: np.ndarray, seed: int,
                 n_boot: int = 2000, n_perm: int = 2000) -> Dict[str, Any]:
    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    s = np.concatenate([member_scores, nonmember_scores])
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)

    boots = []
    for _ in range(n_boot):
        m = rng.choice(member_scores, size=len(member_scores), replace=True)
        n = rng.choice(nonmember_scores, size=len(nonmember_scores), replace=True)
        boots.append(roc_auc_score(np.concatenate([np.ones(len(m)), np.zeros(len(n))]), np.concatenate([m, n])))
    perm_dev = []
    for _ in range(n_perm):
        perm_dev.append(abs(roc_auc_score(rng.permutation(y), s) - 0.5))
    p_value = float((1 + np.sum(np.asarray(perm_dev) >= abs(auc - 0.5))) / (1 + n_perm))

    fpr, tpr, _ = roc_curve(y, s)
    return {
        "auc": auc,
        "auc_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "auc_permutation_p": p_value,
        "advantage_max_tpr_minus_fpr": float(np.max(tpr - fpr)),
        "tpr_at_fpr_0.01": _tpr_at_fpr(y, s, 0.01),
        "tpr_at_fpr_0.1": _tpr_at_fpr(y, s, 0.1),
        "n_members": int(len(member_scores)),
        "n_nonmembers": int(len(nonmember_scores)),
        "member_score_mean": float(np.mean(member_scores)),
        "nonmember_score_mean": float(np.mean(nonmember_scores)),
        "member_score_quantiles": np.quantile(member_scores, [0.1, 0.25, 0.5, 0.75, 0.9]).round(5).tolist(),
        "nonmember_score_quantiles": np.quantile(nonmember_scores, [0.1, 0.25, 0.5, 0.75, 0.9]).round(5).tolist(),
    }


def run_membership_attacks(model, data_bundle, participation: Dict[int, int], device, seed: int) -> Dict[str, Any]:
    ds = data_bundle.full_dataset
    holdout = data_bundle.eval_data["mia_record_holdout_samples"]
    nonmember_users = data_bundle.eval_data["mia_nonmember_user_samples"]

    # (a) client participation
    participated = [c for c, n in participation.items() if n > 0]
    never = [c for c, n in participation.items() if n == 0]
    member_user_scores = [-per_record_losses(model, ds, data_bundle.client_partitions[c], device).mean() for c in participated]
    nonmember_user_scores = [-per_record_losses(model, ds, idxs, device).mean() for idxs in nonmember_users.values()]
    # Training clients never sampled did not contribute either; they are non-members too.
    nonmember_user_scores += [-per_record_losses(model, ds, data_bundle.client_partitions[c], device).mean() for c in never]
    client_level = score_attack(np.asarray(member_user_scores), np.asarray(nonmember_user_scores), seed)
    client_level["n_never_sampled_training_clients"] = len(never)

    # (b) record membership within participating clients
    mem_idx = [i for c in participated for i in data_bundle.client_partitions[c]]
    non_idx = [i for c in participated for i in holdout.get(c, [])]
    record_level = score_attack(-per_record_losses(model, ds, mem_idx, device),
                                -per_record_losses(model, ds, non_idx, device), seed + 1)

    return {"description": ATTACK_DESCRIPTION, "client_participation": client_level, "record_membership": record_level}
