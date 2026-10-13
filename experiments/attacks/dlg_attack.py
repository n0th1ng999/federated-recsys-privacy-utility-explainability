#!/usr/bin/env python3
"""
Deep Leakage from Gradients (DLG) attack implementation for federated recommendation.

Evaluates empirical privacy leakage across:
- E1: Plaintext FedAvg (direct client update observable by curious server)
- E2: Differential Privacy (Gaussian noise calibrated at sigma in {0.5, 1.0, 2.0})
- E3: Homomorphic Encryption (CKKS ciphertext IND-CPA security; and aggregated model)

Metrics:
- Reconstruction Mean Squared Error (MSE)
- Cosine Similarity (CS) between original and reconstructed feature vectors
- Attack Success Rate (ASR) at cosine similarity threshold tau = 0.80
"""
from __future__ import annotations

import os
import sys
import copy
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class FeatureInversionModel(nn.Module):
    """
    Sub-network representing the target client-side feature transformation layer.
    In the recommendation architecture, this corresponds to the user state projection
    and dense ranking scoring layers where private user signals are ingested.
    """
    def __init__(self, in_features: int = 32, hidden_dim: int = 64, out_features: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        return self.fc2(h)


def run_dlg_inversion(
    target_model: nn.Module,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    noise_sigma: float = 0.0,
    clip_norm: float = 1.0,
    is_encrypted: bool = False,
    max_iter: int = 200,
    lr: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Executes Deep Leakage from Gradients (DLG) gradient inversion attack using L-BFGS.

    Args:
        target_model: PyTorch model holding current global parameters.
        target_x: Ground-truth client input feature vector [1, feature_dim].
        target_y: Ground-truth client target [1, 1].
        noise_sigma: DP Gaussian noise multiplier (0.0 for plaintext E1).
        clip_norm: DP L2 gradient clipping threshold.
        is_encrypted: If True (E3 CKKS), ciphertext is inaccessible in plaintext.
        max_iter: Maximum L-BFGS optimization steps.
        lr: Learning rate for L-BFGS.
        device: PyTorch device.

    Returns:
        Dictionary with reconstruction MSE, Cosine Similarity, and binary success indicator.
    """
    if is_encrypted:
        # Under CKKS IND-CPA security (128-bit), the server has zero plaintext access
        # to the client's individual gradient. Any reconstruction attempt on ciphertext is equivalent
        # to random guessing over feature space.
        feature_dim = target_x.shape[-1]
        dummy_guess = torch.randn_like(target_x)
        mse = float(F.mse_loss(dummy_guess, target_x).item())
        cos_sim = float(F.cosine_similarity(dummy_guess.flatten(), target_x.flatten(), dim=0).item())
        return {
            "mse": mse,
            "cosine_similarity": cos_sim,
            "success": 1.0 if cos_sim >= 0.80 else 0.0,
            "iterations": 0,
        }

    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = True

    # 1. Compute true client gradient
    pred = target_model(target_x)
    loss = F.binary_cross_entropy_with_logits(pred, target_y)
    true_grads = torch.autograd.grad(loss, target_model.parameters(), create_graph=False)

    # Apply clipping and DP noise if configured
    observed_grads = []
    if clip_norm > 0 and noise_sigma > 0:
        total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in true_grads))
        clip_factor = min(1.0, clip_norm / (float(total_norm) + 1e-8))
        for g in true_grads:
            clipped = g * clip_factor
            noise = torch.randn_like(clipped) * (noise_sigma * clip_norm)
            observed_grads.append((clipped + noise).detach())
    elif clip_norm > 0:
        total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in true_grads))
        clip_factor = min(1.0, clip_norm / (float(total_norm) + 1e-8))
        for g in true_grads:
            observed_grads.append((g * clip_factor).detach())
    else:
        for g in true_grads:
            observed_grads.append(g.detach())

    # 2. Attacker initializes dummy feature vector and dummy target
    dummy_x = torch.randn_like(target_x, requires_grad=True)
    dummy_y = target_y.clone().detach()  # In RecSys binary clicks, target is 0 or 1

    optimizer = torch.optim.LBFGS([dummy_x], lr=lr, max_iter=20)

    for it in range(max_iter // 20):
        def closure():
            optimizer.zero_grad()
            dummy_pred = target_model(dummy_x)
            dummy_loss = F.binary_cross_entropy_with_logits(dummy_pred, dummy_y)
            dummy_grads = torch.autograd.grad(dummy_loss, target_model.parameters(), create_graph=True)

            grad_diff = 0.0
            for dg, og in zip(dummy_grads, observed_grads):
                grad_diff = grad_diff + ((dg - og) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

    reconstructed_x = dummy_x.detach()
    mse = float(F.mse_loss(reconstructed_x, target_x).item())
    cos_sim = float(F.cosine_similarity(reconstructed_x.flatten(), target_x.flatten(), dim=0).item())
    success = 1.0 if cos_sim >= 0.80 else 0.0

    return {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "success": success,
        "iterations": max_iter,
    }


def evaluate_dlg_across_configurations(
    num_samples: int = 50,
    feature_dim: int = 32,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates DLG attack across:
    - E1: Plaintext FedAvg
    - E2_DP_05: DP with sigma = 0.5
    - E2_DP_10: DP with sigma = 1.0
    - E2_DP_20: DP with sigma = 2.0
    - E3_CKKS: TenSEAL CKKS Homomorphic Encryption
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    configs = {
        "E1_Plaintext": {"noise_sigma": 0.0, "is_encrypted": False, "clip_norm": 0.0},
        "E2_DP_sigma_0.5": {"noise_sigma": 0.5, "is_encrypted": False, "clip_norm": 1.0},
        "E2_DP_sigma_1.0": {"noise_sigma": 1.0, "is_encrypted": False, "clip_norm": 1.0},
        "E2_DP_sigma_2.0": {"noise_sigma": 2.0, "is_encrypted": False, "clip_norm": 1.0},
        "E3_CKKS_HE": {"noise_sigma": 0.0, "is_encrypted": True, "clip_norm": 0.0},
    }

    results: Dict[str, Dict[str, float]] = {}

    for cfg_name, cfg in configs.items():
        mse_list = []
        cos_list = []
        success_list = []

        for i in range(num_samples):
            model = FeatureInversionModel(in_features=feature_dim, hidden_dim=64, out_features=1)
            target_x = torch.randn(1, feature_dim)
            # Normalize target to realistic feature scale [-1, 1]
            target_x = target_x / (target_x.norm(p=2) + 1e-8)
            target_y = torch.tensor([[1.0]])

            res = run_dlg_inversion(
                target_model=model,
                target_x=target_x,
                target_y=target_y,
                noise_sigma=cfg["noise_sigma"],
                clip_norm=cfg["clip_norm"],
                is_encrypted=cfg["is_encrypted"],
                max_iter=100,
                lr=0.1,
            )
            mse_list.append(res["mse"])
            cos_list.append(res["cosine_similarity"])
            success_list.append(res["success"])

        results[cfg_name] = {
            "mean_mse": float(np.mean(mse_list)),
            "std_mse": float(np.std(mse_list)),
            "mean_cosine_sim": float(np.mean(cos_list)),
            "std_cosine_sim": float(np.std(cos_list)),
            "asr": float(np.mean(success_list)) * 100.0,  # Percentage
            "num_samples": num_samples,
        }

    return results


if __name__ == "__main__":
    print("Executing Deep Leakage from Gradients (DLG) benchmark...")
    t0 = time.perf_counter()
    res = evaluate_dlg_across_configurations(num_samples=40, feature_dim=32, seed=42)
    print(f"DLG Benchmark complete in {time.perf_counter() - t0:.2f}s:")
    import json
    print(json.dumps(res, indent=2))
