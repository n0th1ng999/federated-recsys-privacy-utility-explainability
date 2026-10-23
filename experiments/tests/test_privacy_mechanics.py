"""Checks for the privacy mechanics that the manuscript's claims depend on."""
import math

import numpy as np
import pytest
import torch

import dp_accounting
from dp_accounting import rdp

from experiments.federated.dp_engine import DPEngine, compute_epsilon
from experiments.federated.he_engine import HEEngine


def test_epsilon_matches_library_and_is_monotone():
    eps = [compute_epsilon(s / 2.0, 20, 5, 15, 1e-5) for s in (1.0, 2.0, 4.0, 8.0)]
    assert all(a > b for a, b in zip(eps, eps[1:]))
    acc = rdp.RdpAccountant(neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE)
    acc.compose(dp_accounting.SampledWithoutReplacementDpEvent(20, 5, dp_accounting.GaussianDpEvent(2.0)), 15)
    assert eps[2] == pytest.approx(acc.get_epsilon(1e-5), rel=0.02)


def test_engine_uses_replace_one_sensitivity():
    engine = DPEngine(clipping_norm=1.0, noise_multiplier=4.0, target_delta=1e-5)
    for _ in range(15):
        engine.step_accounting(20, 5)
    eps, _ = engine.get_privacy_spent()
    assert eps == pytest.approx(compute_epsilon(2.0, 20, 5, 15, 1e-5))


def test_distributed_noise_sums_to_central_noise():
    m, d, trials = 5, 2000, 200
    engine = DPEngine(clipping_norm=1.0, noise_multiplier=4.0)
    zero = {"w": torch.zeros(d)}
    central = torch.stack([engine.add_gaussian_noise(zero, engine.central_noise_std(m))["w"] for _ in range(trials)])
    distributed = torch.stack([
        sum(engine.add_gaussian_noise(zero, engine.distributed_noise_std(m))["w"] for _ in range(m)) / m
        for _ in range(trials)
    ])
    assert distributed.std().item() == pytest.approx(central.std().item(), rel=0.03)
    assert central.std().item() == pytest.approx(4.0 / m, rel=0.03)


def test_clipping_bounds_norm():
    engine = DPEngine(clipping_norm=1.0)
    clipped, norm = engine.clip_update({"a": torch.full((10,), 3.0), "b": torch.full((5,), -2.0)})
    total = math.sqrt(sum(float(v.norm() ** 2) for v in clipped.values()))
    assert norm > 1.0 and total == pytest.approx(1.0, rel=1e-5)


def test_ckks_aggregate_matches_plaintext_and_server_has_no_secret_key():
    he = HEEngine()
    assert not he.server_context.has_secret_key()
    rng = np.random.default_rng(0)
    vecs = [rng.normal(0, 0.05, size=10000) for _ in range(5)]
    weights = [0.1, 0.2, 0.3, 0.25, 0.15]
    payloads = [he.client_encrypt(v)[0] for v in vecs]
    agg, _, nbytes = he.server_aggregate(payloads, weights)
    dec, _ = he.key_holder_decrypt(agg, 10000)
    expected = sum(w * v for w, v in zip(weights, vecs))
    assert np.max(np.abs(dec - expected)) < 1e-4
    assert nbytes == sum(len(b) for b in agg)
