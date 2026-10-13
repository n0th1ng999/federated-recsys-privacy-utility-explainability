from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def compute_rdp(q: float, sigma: float, alpha: float) -> float:
    """Computes Rényi Differential Privacy (RDP) for subsampled Gaussian mechanism."""
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * (sigma ** 2))
    # Approximation for small q
    return (q ** 2) * alpha / (2.0 * (sigma ** 2))


def rdp_to_dp(rdp_orders: List[float], rdp_values: List[float], delta: float) -> float:
    """Converts RDP guarantees to (epsilon, delta)-DP guarantee."""
    eps_candidates = []
    for alpha, rdp in zip(rdp_orders, rdp_values):
        if alpha > 1.0:
            eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
            eps_candidates.append(eps)
    return float(min(eps_candidates)) if eps_candidates else float("inf")


class DPEngine:
    """
    Implements Client-level Differential Privacy (DP-FedAvg) with update clipping,
    calibrated Gaussian noise addition, and Rényi Differential Privacy accounting.
    """

    def __init__(
        self,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        target_delta: float = 1e-5,
    ):
        self.clipping_norm = float(clipping_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.target_delta = float(target_delta)
        self.steps = 0
        self.rdp_orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.rdp_total = [0.0] * len(self.rdp_orders)

    def clip_update(self, update: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Clips model parameter update dict to L2 norm <= clipping_norm.
        Returns clipped update and original L2 norm.
        """
        total_norm_sq = 0.0
        for tensor in update.values():
            total_norm_sq += float(tensor.detach().float().norm(2).item() ** 2)
        total_norm = math.sqrt(total_norm_sq)

        clip_factor = min(1.0, self.clipping_norm / max(total_norm, 1e-8))
        clipped = {k: v.detach().clone() * clip_factor for k, v in update.items()}
        return clipped, total_norm

    def add_noise_to_aggregated_update(
        self,
        aggregated_update: Dict[str, torch.Tensor],
        num_sampled_clients: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Adds Gaussian noise scaled to clipping_norm * noise_multiplier / num_sampled_clients.
        """
        noisy_update = {}
        # sigma_eff = (noise_multiplier * C) / num_sampled_clients
        noise_std = (self.noise_multiplier * self.clipping_norm) / max(1, num_sampled_clients)

        for key, tensor in aggregated_update.items():
            noise = torch.randn_like(tensor.float(), device=tensor.device) * noise_std
            noisy_update[key] = tensor + noise.to(tensor.dtype)

        return noisy_update

    def step_accounting(self, q: float) -> None:
        """Accumulates privacy loss for one federated round with sampling ratio q."""
        self.steps += 1
        for i, alpha in enumerate(self.rdp_orders):
            self.rdp_total[i] += compute_rdp(q=q, sigma=self.noise_multiplier, alpha=alpha)

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Returns computed (epsilon, delta)."""
        eps = rdp_to_dp(self.rdp_orders, self.rdp_total, self.target_delta)
        return float(eps), float(self.target_delta)
