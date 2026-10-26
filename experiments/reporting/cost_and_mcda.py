"""
Cost model and multi-criteria decision analysis (MCDA) for the manuscript.

Cost model (per training job of T rounds, then scaled to a monthly schedule)
---------------------------------------------------------------------------
Every quantity is a measured byte count or time from results/timing, combined
with explicit unit prices (PRICES). Directions are kept separate because cloud
providers bill them differently:

  * client upload (client -> server): measured ciphertext / plaintext / sparse bytes.
    Inbound transfer to the cloud is not billed by the major providers; it is reported
    as volume, and as a cost borne by clients' own networks only if they pay for it.
  * billable server egress (server -> clients): model broadcasts, Top-K masks and the
    one-off distribution of the public CKKS context.
  * server compute: aggregation + key-holder decryption time, plus, for E0, the full
    centralised training time.
  * client compute: local training + encryption time (borne by data holders).
  * key management: one managed key per job holder for the wrapped CKKS secret key.
  * E0 (centralised): raw-data storage, centralised training compute and the
    (unbilled) ingress of raw data. Its cost is therefore NOT zero.

Prices are list prices used as stated assumptions, not quotations.

MCDA
----
Seven criteria. Four are derived from measurements with an explicit mapping to the
1-5 scale; three are judgement-based and justified in MCDA_JUSTIFICATION. The
composite is a weighted sum. Sensitivity analysis: (i) one-at-a-time +-20% weight
perturbation with renormalisation, (ii) 10,000 Dirichlet weight draws centred on
each persona (concentration 20) and from a flat Dirichlet, reporting how often each
architecture ranks first and the mean Kendall tau to the nominal ranking.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
from scipy import stats

PRICES = {
    "compute_usd_per_hour": 0.34,       # 8 vCPU / 16 GiB on-demand instance (c5.2xlarge class)
    "egress_usd_per_gb": 0.09,          # internet data transfer out, first tier
    "ingress_usd_per_gb": 0.0,          # inbound transfer is not billed
    "storage_usd_per_gb_month": 0.023,  # object storage, standard tier
    "kms_usd_per_key_month": 1.0,       # managed key holding the wrapped CKKS secret key
}
SCHEDULE = {"jobs_per_month": 30, "client_scale": 5.0}  # daily retraining; 100 clients vs 20 simulated
GB = 1024 ** 3


def job_cost(perf: Dict[str, Any], train_sec: float, centralised: bool, raw_data_gb: float = 0.0) -> Dict[str, float]:
    scale = SCHEDULE["client_scale"]
    upload_gb = perf.get("uplink_bytes", 0) * scale / GB
    # The public CKKS context is distributed once per key rotation, not per job (added in monthly_cost).
    egress_gb = (perf.get("downlink_model_bytes", 0) + perf.get("downlink_mask_bytes", 0)) * scale / GB
    if centralised:
        server_sec = train_sec
        client_sec = 0.0
        upload_gb = raw_data_gb
        egress_gb = 0.0
    else:
        server_sec = (perf.get("aggregation_time_sec", 0.0) + perf.get("he_decryption_time_sec", 0.0)) * scale
        client_sec = (perf.get("local_training_time_sec", 0.0) + perf.get("he_encryption_time_sec", 0.0)) * scale
    p = PRICES
    c = {
        "upload_gb": upload_gb,
        "billable_egress_gb": egress_gb,
        "server_compute_h": server_sec / 3600,
        "client_compute_h": client_sec / 3600,
        "usd_egress": egress_gb * p["egress_usd_per_gb"],
        "usd_ingress": upload_gb * p["ingress_usd_per_gb"],
        "usd_server_compute": server_sec / 3600 * p["compute_usd_per_hour"],
        "usd_client_compute": client_sec / 3600 * p["compute_usd_per_hour"],
    }
    return c


def monthly_cost(job: Dict[str, float], uses_he: bool, centralised: bool, raw_data_gb: float,
                 context_gb_per_rotation: float = 0.0) -> Dict[str, float]:
    """Monthly totals; keys are rotated once a month, so the public context is sent once to every client."""
    j = SCHEDULE["jobs_per_month"]
    m = {k: v * j for k, v in job.items()}
    if uses_he:
        m["billable_egress_gb"] += context_gb_per_rotation
        m["usd_egress"] += context_gb_per_rotation * PRICES["egress_usd_per_gb"]
    m["usd_key_management"] = PRICES["kms_usd_per_key_month"] if uses_he else 0.0
    m["usd_storage"] = raw_data_gb * PRICES["storage_usd_per_gb_month"] if centralised else 0.0
    m["usd_operator_total"] = m["usd_egress"] + m["usd_ingress"] + m["usd_server_compute"] + m["usd_key_management"] + m["usd_storage"]
    m["usd_total_incl_clients"] = m["usd_operator_total"] + m["usd_client_compute"]
    return m


# ---------------------------------------------------------------------------
# MCDA
# ---------------------------------------------------------------------------
CRITERIA = [
    "C1_raw_data_stays_local",
    "C2_update_confidentiality_vs_server",
    "C3_formal_output_privacy",
    "C4_ranking_utility",
    "C5_monthly_cost",
    "C6_explanation_fidelity",
    "C7_operational_simplicity",
]

JUDGEMENT_SCORES = {
    # C1: raw data leaves the data holder (1) vs. never leaves (5). Factual, not legal.
    "C1_raw_data_stays_local": {"E0": 1, "E1": 5, "E2": 5, "E3": 5, "E3_topk10": 5, "E4": 5},
    # C2: what the aggregation server learns about an individual update.
    "C2_update_confidentiality_vs_server": {"E0": 1, "E1": 1, "E2": 1, "E3": 4, "E3_topk10": 4, "E4": 4},
    # C7: number of extra components to operate (key management, noise calibration, masks).
    "C7_operational_simplicity": {"E0": 5, "E1": 4, "E2": 3, "E3": 2, "E3_topk10": 2, "E4": 1},
}

MCDA_JUSTIFICATION = {
    "C1_raw_data_stays_local": "E0 pools raw interaction records on the server (1). All federated variants keep records on the client and transmit only model updates (5). This criterion describes data flow, not legal compliance.",
    "C2_update_confidentiality_vs_server": "E0/E1/E2: the server receives plaintext data or plaintext updates (1; E2 adds noise only after aggregation). E3/E3-TopK/E4: the server receives CKKS ciphertexts and holds no secret key (4, not 5: confidentiality is conditional on RLWE hardness, on the server not colluding with the key holder, and the aggregate is still revealed to the key holder; Top-K additionally reveals the shared coordinate set, which the server chooses).",
    "C3_formal_output_privacy": "Derived from the accountant: 1 without DP; with DP, 5 if epsilon <= 1, 4 if <= 3, 3 if <= 8, 2 otherwise (client-level, delta = 1e-5).",
    "C4_ranking_utility": "Derived: linear map of risk-adjusted utility, the lower end of the 95% interval of mean nDCG@10 over seeds, from the random ranking's mean (1) to the pre-cutoff popularity ranking's mean (5). Using the lower bound prevents high between-seed variance under DP noise from scoring as utility.",
    "C5_monthly_cost": "Derived: operator monthly cost mapped on a log scale from the cheapest (5) to the most expensive (1) configuration.",
    "C6_explanation_fidelity": "Derived: linear map of the SHAP-minus-random removal-curve AUC gap (conditional replacement) from its minimum (1) to its maximum (5). E3 Top-10% takes E3's value because explanations were not computed for Top-k runs.",
    "C7_operational_simplicity": "Components to operate beyond plaintext FedAvg: E0 none for federation (5); E1 federation only (4); E2 adds noise calibration and an accountant (3); E3/E3-TopK add key generation, custody and context distribution, and for Top-K a mask protocol (2); E4 adds both (1).",
}

PERSONAS = {
    "Regulated (finance/health)": {"C1_raw_data_stays_local": 0.20, "C2_update_confidentiality_vs_server": 0.20,
                                   "C3_formal_output_privacy": 0.20, "C4_ranking_utility": 0.10,
                                   "C5_monthly_cost": 0.05, "C6_explanation_fidelity": 0.15, "C7_operational_simplicity": 0.10},
    "Consumer retail": {"C1_raw_data_stays_local": 0.10, "C2_update_confidentiality_vs_server": 0.05,
                        "C3_formal_output_privacy": 0.05, "C4_ranking_utility": 0.35,
                        "C5_monthly_cost": 0.20, "C6_explanation_fidelity": 0.10, "C7_operational_simplicity": 0.15},
    "B2B consortium": {"C1_raw_data_stays_local": 0.20, "C2_update_confidentiality_vs_server": 0.20,
                       "C3_formal_output_privacy": 0.10, "C4_ranking_utility": 0.20,
                       "C5_monthly_cost": 0.10, "C6_explanation_fidelity": 0.10, "C7_operational_simplicity": 0.10},
}


def _linear(values: Dict[str, float], lo: float, hi: float) -> Dict[str, float]:
    span = hi - lo if hi > lo else 1.0
    return {k: float(np.clip(1 + 4 * (v - lo) / span, 1, 5)) for k, v in values.items()}


def derived_scores(ndcg: Dict[str, float], random_ndcg: float, popularity_ndcg: float, cost: Dict[str, float],
                   fidelity_gap: Dict[str, float], epsilon: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    eps_score = {}
    for a in ndcg:
        e = epsilon.get(a)
        eps_score[a] = 1.0 if e is None else (5.0 if e <= 1 else 4.0 if e <= 3 else 3.0 if e <= 8 else 2.0)
    logc = {a: math.log10(max(c, 1e-4)) for a, c in cost.items()}
    cost_score = {a: 6 - s for a, s in _linear(logc, min(logc.values()), max(logc.values())).items()}
    return {
        "C3_formal_output_privacy": eps_score,
        "C4_ranking_utility": _linear(ndcg, random_ndcg, popularity_ndcg),
        "C5_monthly_cost": cost_score,
        "C6_explanation_fidelity": _linear(fidelity_gap, min(fidelity_gap.values()), max(fidelity_gap.values())),
    }


def composite(scores: Dict[str, Dict[str, float]], weights: Dict[str, float], archs: List[str]) -> Dict[str, float]:
    return {a: sum(weights[c] * scores[c][a] for c in CRITERIA) for a in archs}


def sensitivity(scores, archs: List[str], seed: int = 0, draws: int = 10000) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    for persona, w in PERSONAS.items():
        nominal = composite(scores, w, archs)
        nominal_rank = sorted(archs, key=lambda a: -nominal[a])
        # one-at-a-time +-20%
        oat = {}
        for c in CRITERIA:
            for f in (0.8, 1.2):
                w2 = dict(w)
                w2[c] *= f
                s = sum(w2.values())
                w2 = {k: v / s for k, v in w2.items()}
                comp = composite(scores, w2, archs)
                oat[f"{c}x{f}"] = sorted(archs, key=lambda a: -comp[a])[0]
        # Dirichlet around persona and flat
        res = {}
        for label, alpha in (("centred", 20 * np.array([w[c] for c in CRITERIA])), ("flat", np.ones(len(CRITERIA)))):
            first = {a: 0 for a in archs}
            taus = []
            W = rng.dirichlet(alpha, size=draws)
            S = np.array([[scores[c][a] for c in CRITERIA] for a in archs])  # archs x criteria
            comp = W @ S.T  # draws x archs
            for row in comp:
                order = np.argsort(-row)
                first[archs[order[0]]] += 1
                taus.append(stats.kendalltau([nominal_rank.index(archs[i]) for i in order], range(len(archs)))[0])
            res[label] = {"p_first": {a: first[a] / draws for a in archs}, "mean_kendall_tau_to_nominal": float(np.mean(taus))}
        out[persona] = {
            "nominal": nominal,
            "nominal_ranking": nominal_rank,
            "oat_top_choice_unchanged": all(v == nominal_rank[0] for v in oat.values()),
            "oat_top_choices": sorted(set(oat.values())),
            "dirichlet": res,
        }
    return out
