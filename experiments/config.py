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


def get_e2_fl_dp_config(
    level: str = "DP-2",
    clipping_norm: float = 1.0,
    seed: int = 42,
) -> ExperimentConfig:
    # Noise multiplier presets matching Section 6:
    # DP-1: Higher epsilon / lower noise / lower privacy
    # DP-2: Medium epsilon / medium noise
    # DP-3: Lower epsilon / higher noise / stronger privacy
    noise_multipliers = {
        "DP-1": 0.5,
        "DP-2": 1.0,
        "DP-3": 2.0,
    }
    sigma = noise_multipliers.get(level, 1.0)
    return ExperimentConfig(
        name=f"E2_FL_{level.replace('-', '')}",
        description=f"Federated Learning with Differential Privacy ({level})",
        is_federated=True,
        dp=DPConfig(
            enabled=True,
            privacy_level=level,
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


def get_e4_fl_dp_he_config(seed: int = 42) -> ExperimentConfig:
    return ExperimentConfig(
        name="E4_FL_DP_HE",
        description="Combined Federated Learning with Differential Privacy (DP-2) and Homomorphic Encryption (CKKS)",
        is_federated=True,
        dp=DPConfig(
            enabled=True,
            privacy_level="DP-2",
            clipping_norm=1.0,
            noise_multiplier=1.0,
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


def get_all_experiment_configs(seed: int = 42) -> Dict[str, ExperimentConfig]:
    return {
        "E0": get_e0_centralized_config(seed=seed),
        "E1": get_e1_fl_baseline_config(seed=seed),
        "E2_DP1": get_e2_fl_dp_config(level="DP-1", seed=seed),
        "E2_DP2": get_e2_fl_dp_config(level="DP-2", seed=seed),
        "E2_DP3": get_e2_fl_dp_config(level="DP-3", seed=seed),
        "E3": get_e3_fl_he_config(seed=seed),
        "E4": get_e4_fl_dp_he_config(seed=seed),
    }


def get_quick_experiment_configs(seed: int = 42) -> Dict[str, ExperimentConfig]:
    """Lightweight configuration suite for rapid verification runs."""
    configs = {
        "E0": get_e0_centralized_config(seed=seed),
        "E1": get_e1_fl_baseline_config(seed=seed),
        "E2_DP2": get_e2_fl_dp_config(level="DP-2", seed=seed),
        "E3": get_e3_fl_he_config(seed=seed),
        "E4": get_e4_fl_dp_he_config(seed=seed),
    }
    for cfg in configs.values():
        cfg.fl.num_clients = 3
        cfg.fl.clients_per_round = 2
        cfg.fl.num_rounds = 2
        cfg.fl.local_epochs = 1
        cfg.eval.eval_users = 5
        cfg.eval.candidates_per_user = 10
        cfg.eval.shap_samples = 5
        cfg.eval.lime_samples = 5
        cfg.eval.stability_runs = 2
    return configs
