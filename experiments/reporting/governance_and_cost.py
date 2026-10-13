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
# 2. Cloud Egress Cost and Operational Carbon Footprint Model
# ---------------------------------------------------------------------------
def compute_cloud_egress_and_carbon(
    num_clients: int = 100,
    cadence_days_per_month: int = 30,
    rounds_per_training: int = 15,
) -> Dict[str, Any]:
    """
    Calculates monthly egress data volume, cloud provider cost (AWS/GCP/Azure standard egress: $0.09/GB),
    and operational carbon emission (based on 0.385 kg CO2e / kWh and datacenter PUE 1.2).
    """
    # Baseline communication volumes per 15-round training (MB):
    # E1 / E2: 167.85 MB
    # E3 / E4: 3,610.90 MB
    # E3 Top-K (K=10%): 344.00 MB
    payloads_mb = {
        "E1_Plaintext": 167.85,
        "E2_DP": 167.85,
        "E3_CKKS_HE_Full": 3610.90,
        "E3_CKKS_TopK_10pct": 344.00,
        "E4_Hybrid_Full": 3610.94,
    }

    cost_per_gb = 0.09  # Tier 1 AWS/GCP standard egress rate ($/GB)
    kwh_per_gb_transfer = 0.005  # Average network transmission energy: ~0.005 kWh/GB
    grid_carbon_intensity = 385.0  # Global average grid intensity: ~385 gCO2e / kWh
    server_watt_hours_per_sec = 150.0 / 3600.0  # 150W server compute draw

    # Latencies per round (server aggregation seconds)
    agg_times_sec = {
        "E1_Plaintext": 0.042 * rounds_per_training,
        "E2_DP": 0.188 * rounds_per_training,
        "E3_CKKS_HE_Full": 24.516,
        "E3_CKKS_TopK_10pct": 2.450,
        "E4_Hybrid_Full": 25.302,
    }

    results = {}
    for arch, mb_per_job in payloads_mb.items():
        total_gb_month = (mb_per_job * (num_clients / 20.0) * cadence_days_per_month) / 1024.0
        egress_cost_usd = total_gb_month * cost_per_gb

        # Compute energy
        server_sec = agg_times_sec[arch] * cadence_days_per_month
        compute_kwh = (server_sec * server_watt_hours_per_sec) / 1000.0
        network_kwh = total_gb_month * kwh_per_gb_transfer
        total_kwh = compute_kwh + network_kwh
        carbon_g = total_kwh * grid_carbon_intensity

        results[arch] = {
            "monthly_transfer_gb": round(total_gb_month, 2),
            "monthly_egress_cost_usd": round(egress_cost_usd, 2),
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

    print("\n--- Monthly Cloud Cost & Carbon (100 clients, 30 days) ---")
    for arch, metrics in cost_carbon.items():
        print(f"  {arch:22s}: {metrics['monthly_transfer_gb']:8.2f} GB | ${metrics['monthly_egress_cost_usd']:6.2f} | {metrics['monthly_carbon_gco2eq']:7.1f} gCO2eq")


if __name__ == "__main__":
    main()
