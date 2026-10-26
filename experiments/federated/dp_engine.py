from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

import dp_accounting
from dp_accounting import rdp, pld


# RDP orders searched by the accountant when converting to (epsilon, delta).
RDP_ORDERS = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 257))


def compute_epsilon(
    noise_multiplier: float,
    num_clients: int,
    clients_per_round: int,
    num_rounds: int,
    delta: float,
) -> float:
    """
    Client-level (epsilon, delta) for DP-FedAvg with fixed-size sampling of
    `clients_per_round` out of `num_clients` WITHOUT replacement, Gaussian noise
    multiplier `noise_multiplier`, composed over `num_rounds` rounds.

    Uses the RDP accountant of Google's dp_accounting library with the
    replace-one neighbouring relation (the relation under which the subsampled
    Gaussian without replacement is analysed; Wang, Balle & Kasiviswanathan, 2019).
    `noise_multiplier` here is the ratio of the noise std to the L2 sensitivity
    of the released quantity (see DPEngine for the 2C sensitivity of the mean).
    The result is an upper bound on epsilon, not an exact value.
    """
    if num_rounds == 0:
        return 0.0
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.SampledWithoutReplacementDpEvent(
            source_dataset_size=num_clients,
            sample_size=clients_per_round,
            event=dp_accounting.GaussianDpEvent(noise_multiplier),
        ),
        num_rounds,
    )
    accountant = rdp.RdpAccountant(
        orders=RDP_ORDERS,
        neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE,
    )
    accountant.compose(event)
    return float(accountant.get_epsilon(delta))


def compute_epsilon_poisson_pld(
    noise_multiplier: float,
    sampling_probability: float,
    num_rounds: int,
    delta: float,
) -> float:
    """
    Reference value only: PLD accountant for Poisson subsampling at the same rate
    (add/remove relation). Reported as a cross-check, not as the guarantee.
    """
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
        ),
        num_rounds,
    )
    accountant = pld.PLDAccountant()
    accountant.compose(event)
    return float(accountant.get_epsilon(delta))


class DPEngine:
    """
    Client-level DP-FedAvg (McMahan et al., 2018). Each sampled client's update is
    clipped to L2 norm C, the server takes the UNIFORM mean over the fixed number m
    of sampled clients, and Gaussian noise with std sigma*C/m is added to that mean.

    Privacy unit: one client. In this study one client holds the data of exactly
    one user, so client-level and user-level DP coincide.

    Neighbouring relation: replace-one (one client's data replaced by arbitrary
    data). Replacing one clipped update changes the mean by at most 2C/m, so the
    noise multiplier relative to the L2 sensitivity is sigma/2. That ratio is
    what the accountant receives.
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
        self.num_clients = None
        self.clients_per_round = None

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

    def central_noise_std(self, num_sampled_clients: int) -> float:
        """Std of the noise added to the uniform mean of m clipped updates."""
        return (self.noise_multiplier * self.clipping_norm) / max(1, num_sampled_clients)

    def distributed_noise_std(self, num_sampled_clients: int) -> float:
        """
        Per-client noise std for distributed DP under secure aggregation (E4):
        each of the m clients adds N(0, (sigma*C)^2 / m) to its clipped update, so the
        SUM carries N(0, (sigma*C)^2) and the uniform mean carries std sigma*C/m,
        exactly the central-DP noise of E2.
        """
        m = max(1, num_sampled_clients)
        return (self.noise_multiplier * self.clipping_norm) / math.sqrt(m)

    def add_gaussian_noise(self, update: Dict[str, torch.Tensor], std: float) -> Dict[str, torch.Tensor]:
        noisy = {}
        for key, tensor in update.items():
            noise = torch.randn_like(tensor.float(), device=tensor.device) * std
            noisy[key] = tensor + noise.to(tensor.dtype)
        return noisy

    def step_accounting(self, num_clients: int, clients_per_round: int) -> None:
        """Records one round of fixed-size sampling (m of N without replacement)."""
        if self.num_clients is None:
            self.num_clients, self.clients_per_round = num_clients, clients_per_round
        elif (self.num_clients, self.clients_per_round) != (num_clients, clients_per_round):
            raise ValueError("Sampling configuration changed between rounds; accountant assumes it is fixed.")
        self.steps += 1

    @property
    def sensitivity_noise_multiplier(self) -> float:
        """Noise std divided by the replace-one L2 sensitivity (2C/m) of the mean."""
        return self.noise_multiplier / 2.0

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Returns the RDP upper bound on (epsilon, delta) after the recorded rounds."""
        if self.steps == 0:
            return 0.0, self.target_delta
        eps = compute_epsilon(
            noise_multiplier=self.sensitivity_noise_multiplier,
            num_clients=self.num_clients,
            clients_per_round=self.clients_per_round,
            num_rounds=self.steps,
            delta=self.target_delta,
        )
        return eps, self.target_delta
