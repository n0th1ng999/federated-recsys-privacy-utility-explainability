#!/usr/bin/env python3
"""
Gradient-inversion (reconstruction) attack on the actual recommender (CoreHybridRecSys).

What is reconstructed: the 43 tabular summary features of one training record
(user-state summaries, collaborative channel support and pair context). The
attacker is given the record's label and its sequence (cluster) inputs, which
makes the attack stronger than realistic and the results an upper bound on leakage.

Update model: one client's update is the gradient of the purchase-head BCE on a
single record at the current global model (batch size 1, the setting most
favourable to the attacker). Real updates (one Adam epoch over many records) are
harder to invert.

Observation scenarios, keyed to who sees what in each configuration:
  individual_plain      E1 server; E2 server (central DP: updates arrive before noise; clipped)
  individual_distnoise  E4 update if CKKS confidentiality failed or server colluded with key holder
                        (noise std sigma*C/sqrt(m))
  aggregate_plain       mean of m clipped updates, no noise: E1 / E3 key holder, or anyone
                        differencing consecutive E1/E3 global models
  aggregate_centralnoise  E2 released model; E4 key holder (noise std sigma*C/m)
  E3/E4 aggregation server: sees only CKKS ciphertexts; gradient inversion is not
  applicable. Its protection rests on RLWE hardness (IND-CPA) and key separation,
  which this experiment does not test.

Attack: gradient matching with cosine distance (Geiping et al., 2020), Adam on the
dummy features in standardised space, initialised at the reference mean. Success is
measured in standardised feature space against a prior-only baseline (guessing the
reference mean), which is what an attacker knows without the update.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss  # noqa: F401
import argparse
import json
import math
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.data.partitioner import prepare_experiment_data
from experiments.evaluation.xai_evaluator import (
    COLLAB_LEN, INTENT_LEN, OWNERSHIP_LEN, extract_tabular_features_from_sample,
)
from recommender_core.Core_Model import CoreHybridRecSys

ATTACK_PARAMS = {
    "optimizer": "Adam", "lr": 0.05, "iterations": 300, "restarts": 1,
    "loss": "1 - cosine(grad_dummy, grad_observed) over all parameters",
    "init": "reference mean (standardised zero)",
    "label_known": True, "sequence_inputs_known": True,
    "clients_per_aggregate": 5, "clip_norm": 1.0,
    "success_threshold_cosine_standardised": 0.5,
}


def _inputs(x: torch.Tensor, sample: Dict[str, Any]) -> Dict[str, Any]:
    def seq(v):
        t = torch.as_tensor(np.asarray(v)).float()
        return t.reshape(1, t.shape[-2], t.shape[-1])
    a, b, c = INTENT_LEN, INTENT_LEN + OWNERSHIP_LEN, INTENT_LEN + OWNERSHIP_LEN + COLLAB_LEN
    return {
        "user_state": {"intent_summary": x[:, :a], "ownership_summary": x[:, a:b],
                       "intent_clusters": seq(sample["user_state"]["intent_clusters"]),
                       "ownership_clusters": seq(sample["user_state"]["ownership_clusters"])},
        "collaborative": {"channel_support": x[:, b:c], "clusters": seq(sample["collaborative"]["clusters"])},
        "pair_context": {"context": x[:, c:]},
    }


def record_gradient(model, x: torch.Tensor, sample, label: float, create_graph=False) -> List[torch.Tensor]:
    logit, _, _ = model(_inputs(x, sample))
    loss = F.binary_cross_entropy_with_logits(logit.flatten(), torch.tensor([label]))
    return list(torch.autograd.grad(loss, [p for p in model.parameters()], create_graph=create_graph, allow_unused=True))


def _fill(grads, params):
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def _clip(grads, c):
    norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
    f = min(1.0, c / max(norm, 1e-12))
    return [g * f for g in grads]


def invert(model, observed, sample, label, mu, sd, iters, lr, seed) -> np.ndarray:
    torch.manual_seed(seed)
    params = list(model.parameters())
    z = torch.zeros(1, mu.shape[0], requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    obs_flat = torch.cat([g.flatten() for g in observed])
    for _ in range(iters):
        opt.zero_grad()
        x = z * sd + mu
        g = _fill(record_gradient(model, x, sample, label, create_graph=True), params)
        loss = 1 - F.cosine_similarity(torch.cat([t.flatten() for t in g]), obs_flat, dim=0)
        loss.backward()
        opt.step()
    return (z.detach() * sd + mu).numpy().reshape(-1)


def run_seed(seed: int, n_records: int, sigmas: List[float], data_dir: str, iters: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    bundle = prepare_experiment_data(data_dir=data_dir, num_clients=20, eval_users_count=50, seed=seed)
    ds = bundle.full_dataset
    model = CoreHybridRecSys(branch_dims=bundle.branch_dims)
    model.eval()  # deterministic forward pass (no dropout) for both the client and the attacker
    params = list(model.parameters())

    pool = [i for idxs in bundle.client_partitions.values() for i in idxs]
    ref = np.stack([extract_tabular_features_from_sample(ds[int(i)]) for i in rng.choice(pool, 300, replace=False)])
    mu_np, sd_np = ref.mean(0), ref.std(0) + 1e-6
    mu, sd = torch.tensor(mu_np).float(), torch.tensor(sd_np).float()
    m, C = ATTACK_PARAMS["clients_per_aggregate"], ATTACK_PARAMS["clip_norm"]

    clients = list(bundle.client_partitions)
    targets = []
    for _ in range(n_records):
        cs = rng.choice(clients, size=m, replace=False)
        recs = [int(rng.choice(bundle.client_partitions[c])) for c in cs]
        targets.append(recs)  # recs[0] is the target, the rest are co-aggregated clients

    def grad_of(idx):
        s = ds[idx]
        x = torch.tensor(extract_tabular_features_from_sample(s)).float().reshape(1, -1)
        return s, x, float(s["label_reg"]), _clip(_fill(record_gradient(model, x, s, float(s["label_reg"])), params), C)

    scenarios = {"individual_plain": None, "aggregate_plain": None}
    for sg in sigmas:
        scenarios[f"individual_distnoise_s{sg:g}"] = sg
        scenarios[f"aggregate_centralnoise_s{sg:g}"] = sg

    results = {k: {"cos_std": [], "mse_std": [], "baseline_cos_std": [], "baseline_mse_std": []} for k in scenarios}
    t0 = time.perf_counter()
    for r, recs in enumerate(targets):
        s, x_true, y, g_target = grad_of(recs[0])
        others = [grad_of(i)[3] for i in recs[1:]]
        z_true = (x_true.numpy().reshape(-1) - mu_np) / sd_np
        for name, sg in scenarios.items():
            if name == "individual_plain":
                obs = g_target
            elif name.startswith("individual_distnoise"):
                obs = [g + torch.randn_like(g) * sg * C / math.sqrt(m) for g in g_target]
            elif name == "aggregate_plain":
                obs = [sum(gs) / m for gs in zip(g_target, *others)]
            else:
                obs = [sum(gs) / m + torch.randn_like(gs[0]) * sg * C / m for gs in zip(g_target, *others)]
            x_hat = invert(model, obs, s, y, mu, sd, iters, ATTACK_PARAMS["lr"], seed * 1000 + r)
            z_hat = (x_hat - mu_np) / sd_np
            denom = np.linalg.norm(z_true) * np.linalg.norm(z_hat)
            results[name]["cos_std"].append(float(z_true @ z_hat / denom) if denom > 0 else 0.0)
            results[name]["mse_std"].append(float(np.mean((z_hat - z_true) ** 2)))
            results[name]["baseline_cos_std"].append(0.0)  # mean guess is the zero vector in z-space
            results[name]["baseline_mse_std"].append(float(np.mean(z_true ** 2)))
        print(f"  seed {seed} record {r + 1}/{n_records} ({time.perf_counter() - t0:.0f}s)", flush=True)

    thr = ATTACK_PARAMS["success_threshold_cosine_standardised"]
    summary = {}
    for name, v in results.items():
        cos = np.asarray(v["cos_std"])
        mse = np.asarray(v["mse_std"])
        base = np.asarray(v["baseline_mse_std"])
        summary[name] = {
            "mean_cos_std": float(cos.mean()),
            "median_cos_std": float(np.median(cos)),
            "success_rate": float(np.mean(cos >= thr)),
            "mean_mse_std": float(mse.mean()),
            "mse_relative_to_prior_guess": float(mse.mean() / base.mean()),
            "per_record": v,
        }
    return {"seed": seed, "n_records": n_records, "params": ATTACK_PARAMS, "sigmas": sigmas, "scenarios": summary,
            "not_applicable": {"E3_server": "ciphertexts only (CKKS)", "E4_server": "ciphertexts only (CKKS)"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--records", type=int, default=20)
    ap.add_argument("--iters", type=int, default=ATTACK_PARAMS["iterations"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=os.path.join("experiments", "results", "main", "attacks"))
    args = ap.parse_args()
    ATTACK_PARAMS["iterations"] = args.iters
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1)
    for seed in args.seeds:
        path = os.path.join(args.out, f"reconstruction_seed_{seed}.json")
        if os.path.exists(path):
            continue
        res = run_seed(seed, args.records, args.sigmas, args.data_dir, args.iters)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
