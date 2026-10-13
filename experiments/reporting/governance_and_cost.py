#!/usr/bin/env python3
"""
Enterprise Governance Scoring Rubric (MCDA) and Cloud Egress Cost / Carbon Footprint Model
for Privacy-Preserving Federated Recommender Systems.

Produces:
1. Multi-Criteria Decision Analysis (MCDA) scores across 6 enterprise governance dimensions
   under three organizational personas (Financial/Healthcare, Consumer Retail, B2B Consortium).
2. Cloud egress network transfer cost and operational carbon emission (gCO2eq) estimates.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Multi-Criteria Enterprise Governance Scoring Rubric
# ---------------------------------------------------------------------------
# Dimensions scored on a 1 (Unfavorable / Low) to 5 (Optimal / High) Likert scale:
# D1: Regulatory Compliance (GDPR Art. 25, HIPAA, CCPA)
# D2: Confidentiality Guarantee (Information-Theoretic / Cryptographic / Statistical)
# D3: Explainability & Auditability (Attribution stability and fidelity)
# D4: Systems Infrastructure Feasibility (Bandwidth and memory friendliness)
# D5: Total Cost of Ownership (Compute & cloud network cost)
# D6: Operational Simplicity (Key management and calibration friction)

CRITERIA = [
    "D1_Regulatory_Compliance",
    "D2_Confidentiality_Guarantee",
    "D3_Explainability_Auditability",
    "D4_Infrastructure_Feasibility",
    "D5_TCO_Cost_Efficiency",
    "D6_Operational_Simplicity",
]

ARCHITECTURES = {
    "E0_Centralized": {
        "D1_Regulatory_Compliance": 1,
        "D2_Confidentiality_Guarantee": 1,
        "D3_Explainability_Auditability": 5,
        "D4_Infrastructure_Feasibility": 5,
        "D5_TCO_Cost_Efficiency": 5,
        "D6_Operational_Simplicity": 5,
    },
    "E1_Plaintext_FL": {
        "D1_Regulatory_Compliance": 3,
        "D2_Confidentiality_Guarantee": 2,
        "D3_Explainability_Auditability": 4,
        "D4_Infrastructure_Feasibility": 4,
        "D5_TCO_Cost_Efficiency": 4,
        "D6_Operational_Simplicity": 4,
    },
    "E2_DP_FL": {
        "D1_Regulatory_Compliance": 5,
        "D2_Confidentiality_Guarantee": 4,
        "D3_Explainability_Auditability": 2,
        "D4_Infrastructure_Feasibility": 4,
        "D5_TCO_Cost_Efficiency": 4,
        "D6_Operational_Simplicity": 3,
    },
    "E3_CKKS_HE_FL": {
        "D1_Regulatory_Compliance": 4,
        "D2_Confidentiality_Guarantee": 5,
        "D3_Explainability_Auditability": 4,
        "D4_Infrastructure_Feasibility": 1,
        "D5_TCO_Cost_Efficiency": 2,
        "D6_Operational_Simplicity": 2,
    },
    "E4_Hybrid_DP_HE_FL": {
        "D1_Regulatory_Compliance": 5,
        "D2_Confidentiality_Guarantee": 5,
        "D3_Explainability_Auditability": 2,
        "D4_Infrastructure_Feasibility": 1,
        "D5_TCO_Cost_Efficiency": 1,
        "D6_Operational_Simplicity": 1,
    },
}

# Personas with normalized weights across the 6 criteria:
# Persona A: Financial / Healthcare (Confidentiality & Compliance driven)
# Persona B: Consumer Retail (Utility, Cost & Infrastructure driven)
# Persona C: Multi-Enterprise Consortium (Balanced Governance)
PERSONAS = {
    "Persona_A_Regulated_Finance_Health": {
        "D1_Regulatory_Compliance": 0.30,
        "D2_Confidentiality_Guarantee": 0.35,
        "D3_Explainability_Auditability": 0.15,
        "D4_Infrastructure_Feasibility": 0.05,
        "D5_TCO_Cost_Efficiency": 0.10,
        "D6_Operational_Simplicity": 0.05,
    },
    "Persona_B_Consumer_Retail_Commerce": {
        "D1_Regulatory_Compliance": 0.10,
        "D2_Confidentiality_Guarantee": 0.10,
        "D3_Explainability_Auditability": 0.20,
        "D4_Infrastructure_Feasibility": 0.25,
        "D5_TCO_Cost_Efficiency": 0.25,
        "D6_Operational_Simplicity": 0.10,
    },
    "Persona_C_B2B_Enterprise_Consortium": {
        "D1_Regulatory_Compliance": 0.20,
        "D2_Confidentiality_Guarantee": 0.20,
        "D3_Explainability_Auditability": 0.20,
        "D4_Infrastructure_Feasibility": 0.15,
        "D5_TCO_Cost_Efficiency": 0.15,
        "D6_Operational_Simplicity": 0.10,
    },
}


def compute_mcda_rankings() -> Dict[str, Any]:
    scores_by_persona: Dict[str, Dict[str, float]] = {p: {} for p in PERSONAS}
    for p_name, weights in PERSONAS.items():
        for arch, scores in ARCHITECTURES.items():
            composite = sum(scores[crit] * weights[crit] for crit in CRITERIA)
            scores_by_persona[p_name][arch] = round(composite, 3)

    return {
        "raw_scores": ARCHITECTURES,
        "persona_weights": PERSONAS,
        "composite_scores": scores_by_persona,
    }


# ---------------------------------------------------------------------------
# 2. Literature-Grounded Cloud TCO and Operational Carbon Footprint Model
# ---------------------------------------------------------------------------
def compute_cloud_egress_and_carbon(
    num_clients: int = 100,
    cadence_days_per_month: int = 30,
    batch_clients: int = 20,
) -> Dict[str, Any]:
    """
    Calculates monthly compute runtime, cloud provider compute cost (AWS c5.2xlarge @ $0.340/hr),
    monthly egress data volume, network egress cost (AWS/GCP/Azure standard egress: $0.090/GB),
    total monthly TCO, and operational greenhouse gas carbon footprint (PUE 1.2, 150W server draw,
    0.005 kWh/GB transmission energy, and 385 gCO2e/kWh grid intensity).

    Literature Grounding:
    1. Cloud TCO Architecture: Armbrust et al. (2010) CACM & Bonawitz et al. (2019) SysML.
       TCO = C_compute + C_egress
    2. Datacenter & Transmission Carbon Accounting: Patterson et al. (2021) CACM & Aslan et al. (2018).
       Carbon = (E_compute + E_net) * I_grid
    """
    # Baseline communication volumes per 15-round training job (MB):
    # E0: Centralized (0 MB transmitted over WAN)
    # E1 / E2: 167.85 MB
    # E3 / E4: 3,610.90 MB
    # E3 Top-K (K=10%): 344.00 MB
    payloads_mb = {
        "E0_Centralized": 0.0,
        "E1_Plaintext": 167.85,
        "E2_DP": 167.85,
        "E3_CKKS_HE_Full": 3610.90,
        "E3_CKKS_TopK_10pct": 344.00,
        "E4_Hybrid_Full": 3610.94,
    }

    # Dedicated server aggregation time per 15-round job (seconds) measured on benchmark
    agg_times_sec = {
        "E0_Centralized": 0.0,
        "E1_Plaintext": 0.042,
        "E2_DP": 0.188,
        "E3_CKKS_HE_Full": 24.516,
        "E3_CKKS_TopK_10pct": 2.450,
        "E4_Hybrid_Full": 25.302,
    }

    client_scaling = num_clients / float(batch_clients)  # 100 / 20 = 5.0

    # FinOps and Environmental Constants
    compute_hourly_rate = 0.340  # Dedicated AWS EC2 c5.2xlarge ($0.340 / hour)
    cost_per_gb_egress = 0.090  # Tier 1 AWS/GCP/Azure standard Internet egress ($0.090 / GB)
    pue = 1.2  # Datacenter Power Usage Effectiveness (Patterson et al., 2021)
    server_power_w = 150.0  # Enterprise server power draw (150 Watts)
    kwh_per_gb_transfer = 0.005  # Transmission electricity intensity: 0.005 kWh/GB (Aslan et al., 2018)
    grid_carbon_intensity = 385.0  # Global average grid intensity: 385 gCO2e / kWh (IEA, 2022)

    results = {}
    for arch in payloads_mb:
        mb_per_job = payloads_mb[arch]
        t_agg_job = agg_times_sec[arch]

        # Monthly compute latency and cost
        monthly_compute_sec = t_agg_job * client_scaling * cadence_days_per_month
        monthly_compute_cost_usd = (monthly_compute_sec / 3600.0) * compute_hourly_rate

        # Monthly network egress volume and cost
        monthly_transfer_gb = (mb_per_job * client_scaling * cadence_days_per_month) / 1024.0
        monthly_egress_cost_usd = monthly_transfer_gb * cost_per_gb_egress

        # Total Monthly TCO
        monthly_tco_usd = monthly_compute_cost_usd + monthly_egress_cost_usd

        # Operational Energy (kWh) and Carbon Footprint (gCO2eq)
        compute_kwh = (monthly_compute_sec * server_power_w * pue) / (3600.0 * 1000.0)
        network_kwh = monthly_transfer_gb * kwh_per_gb_transfer
        total_kwh = compute_kwh + network_kwh
        carbon_g = total_kwh * grid_carbon_intensity

        results[arch] = {
            "monthly_compute_sec": round(monthly_compute_sec, 2),
            "monthly_compute_cost_usd": round(monthly_compute_cost_usd, 4),
            "monthly_transfer_gb": round(monthly_transfer_gb, 2),
            "monthly_egress_cost_usd": round(monthly_egress_cost_usd, 2),
            "monthly_tco_usd": round(monthly_tco_usd, 2),
            "monthly_energy_kwh": round(total_kwh, 4),
            "monthly_carbon_gco2eq": round(carbon_g, 2),
        }

    return results


def main():
    mcda = compute_mcda_rankings()
    cost_carbon = compute_cloud_egress_and_carbon()

    out_dir = "experiments/results/governance"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "governance_and_cost_model.json")

    combined = {
        "mcda_governance": mcda,
        "cloud_cost_and_carbon": cost_carbon,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Governance rubric and Cloud Cost model saved to {out_file}:")
    print("\n--- Composite MCDA Scores across Personas ---")
    for persona, scores in mcda["composite_scores"].items():
        print(f"\n{persona}:")
        for arch, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {arch:25s}: {score:.3f}")

    print("\n--- Monthly Cloud TCO Breakdown & Carbon (100 clients, 30 days) ---")
    print(f"  {'Architecture':22s} | {'Compute':>8s} | {'Egress':>8s} | {'Total TCO':>10s} | {'Transfer':>9s} | {'Carbon':>10s}")
    print("  " + "-" * 75)
    for arch, metrics in cost_carbon.items():
        print(
            f"  {arch:22s} | "
            f"${metrics['monthly_compute_cost_usd']:7.4f} | "
            f"${metrics['monthly_egress_cost_usd']:7.2f} | "
            f"${metrics['monthly_tco_usd']:9.2f} | "
            f"{metrics['monthly_transfer_gb']:7.2f}GB | "
            f"{metrics['monthly_carbon_gco2eq']:7.1f} g"
        )


if __name__ == "__main__":
    main()
