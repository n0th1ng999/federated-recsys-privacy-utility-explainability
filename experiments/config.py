from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DPConfig:
    enabled: bool = False
    privacy_level: str = "none"  # "DP-1" (low privacy), "DP-2" (medium), "DP-3" (high privacy)
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_delta: float = 1e-5
    computed_epsilon: Optional[float] = None


@dataclass
class HEConfig:
    enabled: bool = False
    scheme: str = "CKKS"
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40


@dataclass
class FLConfig:
    num_clients: int = 20
    clients_per_round: int = 5
    num_rounds: int = 15
    local_epochs: int = 1
    learning_rate: float = 0.001
    batch_size: int = 64
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    # Top-K sparsification of client updates.
    #   "none": dense updates
    #   "shared_mask": all clients keep the SAME k coordinates, chosen by the server
    #       (round 1: seeded random; round t: largest |global update| of round t-1),
    #       so packed CKKS vectors are aligned and can be summed homomorphically.
    #   "per_client": each client keeps its own top-k by magnitude and sends
    #       (value, int32 index) pairs in plaintext (not compatible with HE).
    topk_mode: str = "none"
    topk_ratio: float = 1.0


@dataclass
class EvalConfig:
    eval_users: int = 50
    candidates_per_user: int = 50
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    primary_k: int = 10
    shap_samples: int = 30
    lime_samples: int = 30
    fidelity_top_k: int = 3
    stability_runs: int = 3
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str
    description: str
    is_federated: bool = True
    dp: DPConfig = field(default_factory=DPConfig)
    he: HEConfig = field(default_factory=HEConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# -------------------------------------------------------------
# Standard Experimental Configurations (E0 through E4)
# -------------------------------------------------------------

def get_e0_centralized_config(seed: int = 42) -> ExperimentConfig:
    return ExperimentConfig(
        name="E0_Centralized",
        description="Centralized baseline without Federated Learning, DP, or HE",
        is_federated=False,
        dp=DPConfig(enabled=False),
        he=HEConfig(enabled=False),
        fl=FLConfig(seed=seed),
        eval=EvalConfig(seed=seed),
    )


def get_e1_fl_baseline_config(seed: int = 42) -> ExperimentConfig:
    return ExperimentConfig(
        name="E1_FL_Baseline",
        description="Federated Learning baseline (FedAvg) without DP or HE",
        is_federated=True,
        dp=DPConfig(enabled=False),
        he=HEConfig(enabled=False),
        fl=FLConfig(seed=seed),
        eval=EvalConfig(seed=seed),
    )


# Noise multipliers for the central-DP sweep (E2) and for E4. Epsilon is not fixed here:
# it is computed by the accountant in federated/dp_engine.py for the realised schedule.
E2_NOISE_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)
E4_NOISE_MULTIPLIER = 4.0


def _sigma_tag(sigma: float) -> str:
    return f"s{sigma:g}".replace(".", "p")


def get_e2_fl_dp_config(
    sigma: float = 4.0,
    clipping_norm: float = 1.0,
    seed: int = 42,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"E2_FL_DP_{_sigma_tag(sigma)}",
        description=f"Federated Learning with central client-level DP (sigma={sigma:g})",
        is_federated=True,
        dp=DPConfig(
            enabled=True,
            privacy_level=f"sigma={sigma:g}",
            clipping_norm=clipping_norm,
            noise_multiplier=sigma,
            target_delta=1e-5,
        ),
        he=HEConfig(enabled=False),
        fl=FLConfig(seed=seed),
        eval=EvalConfig(seed=seed),
    )


def get_e3_fl_he_config(seed: int = 42) -> ExperimentConfig:
    return ExperimentConfig(
        name="E3_FL_HE",
        description="Federated Learning with Homomorphic Encryption (CKKS)",
        is_federated=True,
        dp=DPConfig(enabled=False),
        he=HEConfig(
            enabled=True,
            scheme="CKKS",
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale_bits=40,
        ),
        fl=FLConfig(seed=seed),
        eval=EvalConfig(seed=seed),
    )


def get_e4_fl_dp_he_config(seed: int = 42, sigma: float = E4_NOISE_MULTIPLIER) -> ExperimentConfig:
    return ExperimentConfig(
        name="E4_FL_DP_HE",
        description=f"Distributed client-level DP (sigma={sigma:g}) under CKKS secure aggregation",
        is_federated=True,
        dp=DPConfig(
            enabled=True,
            privacy_level=f"sigma={sigma:g}",
            clipping_norm=1.0,
            noise_multiplier=sigma,
            target_delta=1e-5,
        ),
        he=HEConfig(
            enabled=True,
            scheme="CKKS",
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale_bits=40,
        ),
        fl=FLConfig(seed=seed),
        eval=EvalConfig(seed=seed),
    )


TOPK_RATIOS = (0.5, 0.2, 0.1, 0.05)


def get_topk_configs(seed: int = 42) -> Dict[str, ExperimentConfig]:
    """Top-K variants: shared-mask under CKKS (E3) and per-client plaintext Top-K (E1)."""
    configs = {}
    for r in TOPK_RATIOS:
        tag = f"topk{int(round(r * 100)):02d}"
        he_cfg = get_e3_fl_he_config(seed=seed)
        he_cfg.name, he_cfg.description = f"E3_{tag}", f"CKKS secure aggregation with shared-mask Top-K ({r:.0%} of coordinates)"
        he_cfg.fl.topk_mode, he_cfg.fl.topk_ratio = "shared_mask", r
        configs[f"E3_{tag}"] = he_cfg
        pl_cfg = get_e1_fl_baseline_config(seed=seed)
        pl_cfg.name, pl_cfg.description = f"E1_{tag}", f"Plaintext FedAvg with per-client Top-K ({r:.0%} of coordinates)"
        pl_cfg.fl.topk_mode, pl_cfg.fl.topk_ratio = "per_client", r
        configs[f"E1_{tag}"] = pl_cfg
    return configs


def get_all_experiment_configs(seed: int = 42) -> Dict[str, ExperimentConfig]:
    configs = {
        "E0": get_e0_centralized_config(seed=seed),
        "E1": get_e1_fl_baseline_config(seed=seed),
    }
    for sigma in E2_NOISE_MULTIPLIERS:
        configs[f"E2_{_sigma_tag(sigma)}"] = get_e2_fl_dp_config(sigma=sigma, seed=seed)
    configs["E3"] = get_e3_fl_he_config(seed=seed)
    configs["E4"] = get_e4_fl_dp_he_config(seed=seed)
    configs.update(get_topk_configs(seed=seed))
    return configs


def get_quick_experiment_configs(seed: int = 42) -> Dict[str, ExperimentConfig]:
    """Lightweight configuration suite for rapid verification runs."""
    configs = {
        "E0": get_e0_centralized_config(seed=seed),
        "E1": get_e1_fl_baseline_config(seed=seed),
        "E2_s4": get_e2_fl_dp_config(sigma=4.0, seed=seed),
        "E3": get_e3_fl_he_config(seed=seed),
        "E4": get_e4_fl_dp_he_config(seed=seed),
    }
    topk = get_topk_configs(seed=seed)
    configs.update({k: topk[k] for k in ("E3_topk10", "E1_topk10")})
    for cfg in configs.values():
        cfg.fl.num_clients = 4
        cfg.fl.clients_per_round = 2
        cfg.fl.num_rounds = 2
        cfg.fl.local_epochs = 1
        cfg.eval.eval_users = 5
        cfg.eval.candidates_per_user = 10
        cfg.eval.shap_samples = 3
        cfg.eval.lime_samples = 3
        cfg.eval.stability_runs = 2
    return configs
