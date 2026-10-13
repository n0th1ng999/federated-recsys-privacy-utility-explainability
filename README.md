# Privacy, Utility, and Explainability Trade-offs in Privacy-Preserving Federated Recommender Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TenSEAL CKKS](https://img.shields.io/badge/TenSEAL-0.3.16-brightgreen.svg)](https://github.com/OpenMined/TenSEAL)
[![Target Journal](https://img.shields.io/badge/Target_Journal-Data_Technologies_and_Applications-purple.svg)](https://www.emerald.com/insight/publication/issn/2514-9342)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the **official, self-contained reproducibility package** for the paper:  
> **"Privacy, Utility, and Explainability Trade-offs in Privacy-Preserving Federated Recommender Systems"**  
> *Target Journal:* **Data Technologies and Applications (Emerald DTA)**

---

## 📌 Repository Overview

This package contains everything required to independently audit, benchmark, and reproduce all empirical results, privacy defense evaluations, communication mitigations, and explainability benchmarks across both benchmark datasets (**EARS** and **MovieLens-100k**).

### Evaluated Configurations
* **E0:** Centralized Learning (Full Data Colocation)
* **E1:** Plaintext Federated Learning (FedAvg, 5 clients, 15 rounds)
* **E2_DP1:** Federated Learning with DP ($\sigma = 0.5, \epsilon \approx 11.17$)
* **E2_DP2:** Federated Learning with DP ($\sigma = 1.0, \epsilon \approx 5.12$)
* **E2_DP3:** Federated Learning with DP ($\sigma = 2.0, \epsilon \approx 2.44$)
* **E3:** Federated Learning with TenSEAL CKKS Homomorphic Encryption (HE)
* **E4:** Hybrid Defense: Federated Learning with Differential Privacy + Homomorphic Encryption (DP+HE)

---

## 🛠️ Environment Installation

We recommend using **Conda** with Python 3.10.

### Option A: Using Conda (Recommended)
```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate recsys-reproducibility
```

### Option B: Using Standard Python `venv`
```bash
# 1. Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip and install pinned requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quickstart Reproduction

The repository includes a master CLI runner ([`reproduce.py`](reproduce.py)) to reproduce all findings easily.

### 1. Instant Manuscript Figures Regeneration (< 5 seconds)
Regenerate all 8 publication figures directly from the multi-seed empirical benchmark logs:
```bash
python reproduce.py --mode figures
```
*Output:* Renders high-resolution PNG and PDF figures to [`Paper/figures/`](Paper/figures/).

### 2. Instant Tables Regeneration
Recompute all LaTeX and Markdown summary tables:
```bash
python reproduce.py --mode tables
```
*Output:* Formats Tables into `experiments/results/multiseed_benchmark/tables/`.

### 3. Fast Smoke Test (~60 seconds)
Run a micro-benchmark (2 rounds, 5 clients) across centralized, FedAvg, DP, and TenSEAL CKKS to confirm that PyTorch, Flower/Federated engine, and homomorphic encryption work end-to-end on your hardware:
```bash
python reproduce.py --mode smoke
```

### 4. Empirical Privacy Defense Attacks (MIA & DLG Inversion)
Replicate the Membership Inference Attack (MIA) shadow-model attack and Deep Leakage from Gradients (DLG) reconstruction attack (Table IX & Figure 8):
```bash
python reproduce.py --mode attacks
```

### 5. TenSEAL Top-$K$ Parameter Sparsification Sweep
Run the communication sparsification benchmark ($K \in \{100\%, 50\%, 20\%, 10\%, 5\%\}$) demonstrating $90\%$ bandwidth reduction with $125\%$ NDCG retention (Table X & Figure 7):
```bash
python reproduce.py --mode sparsification
```

### 6. Advanced XAI Faithfulness & Agreement Benchmark
Run the marginal feature-permutation masking, faithfulness deletion AUC, and SHAP–LIME rank-agreement analysis (Table VII Panel B):
```bash
python reproduce.py --mode xai
```

### 7. Manuscript Integrity Validation
Verify that all 33 Harvard citations and 18 cross-references in the LaTeX manuscript are balanced and resolved:
```bash
python reproduce.py --mode validate
```

---

## 📊 Paper-to-Code Traceability Matrix

Every figure and table in the manuscript directly maps to its generating code and data:

| Manuscript Item | Description | Generating Script / Module | Source Data File |
| :--- | :--- | :--- | :--- |
| **Table I** | Related Work Positioning | [`Paper/tables.tex`](Paper/tables.tex) | Conceptual synthesis |
| **Table II** | Multi-Adversary Threat Model | [`Paper/tables.tex`](Paper/tables.tex) | Cryptographic assumptions |
| **Table III** | Experimental Hyperparameters | [`experiments/config.py`](experiments/config.py) | Configuration contracts |
| **Table IV** | Recommendation Accuracy (10 Seeds) | [`experiments/reporting/table_generator.py`](experiments/reporting/table_generator.py) | `10seeds_results.json` |
| **Table V** | Computational Overhead & RDP Budget | [`experiments/reporting/table_generator.py`](experiments/reporting/table_generator.py) | `aggregated_summary.json` |
| **Table VI** | Communication Expansion Breakdown | [`experiments/reporting/table_generator.py`](experiments/reporting/table_generator.py) | `aggregated_summary.json` |
| **Table VII** | XAI Metrics & Advanced Faithfulness | [`experiments/evaluation/run_advanced_xai_benchmark.py`](experiments/evaluation/run_advanced_xai_benchmark.py) | `advanced_xai_benchmark.json` |
| **Table VIII** | Cross-Dataset Validation (MovieLens) | [`experiments/reporting/table_generator.py`](experiments/reporting/table_generator.py) | `movielens/aggregated_summary.json` |
| **Table IX** | Privacy Attack Defenses (MIA / DLG) | [`experiments/attacks/mia_attack.py`](experiments/attacks/mia_attack.py) | `mia_benchmark_results.json` |
| **Table X** | Top-$K$ Parameter Sparsification | [`experiments/federated/run_topk_benchmark.py`](experiments/federated/run_topk_benchmark.py) | `topk_sparsification_results.json` |
| **Table XI** | Cloud TCO & MCDA Enterprise Model | [`experiments/reporting/governance_and_cost.py`](experiments/reporting/governance_and_cost.py) | `governance_and_cost_model.json` |
| **Table XII** | Practitioner Decision Guide | [`Paper/tables.tex`](Paper/tables.tex) | Multi-criteria synthesis |
| **Figure 1** | NDCG@10 Ranking Accuracy by Config | [`experiments/reporting/plot_generator.py`](experiments/reporting/plot_generator.py) | `10seeds_results.json` |
| **Figure 2** | Recall@10 Ranking Accuracy by Config | [`experiments/reporting/plot_generator.py`](experiments/reporting/plot_generator.py) | `10seeds_results.json` |
| **Figure 3** | Training & Aggregation Overhead | [`experiments/reporting/plot_generator.py`](experiments/reporting/plot_generator.py) | `aggregated_summary.json` |
| **Figure 4** | SHAP vs. LIME Generation Runtime | [`experiments/reporting/plot_generator.py`](experiments/reporting/plot_generator.py) | `aggregated_summary.json` |
| **Figure 5** | Top-3 Masking Sensitivity & Stability | [`experiments/reporting/plot_generator.py`](experiments/reporting/plot_generator.py) | `aggregated_summary.json` |
| **Figure 6** | Privacy–Utility–XAI Pareto Bubble Chart | [`experiments/reporting/plot_generator.py`](experiments/reporting/plot_generator.py) | `aggregated_summary.json` |
| **Figure 7** | Top-$K$ Sparsification Pareto Frontier | [`experiments/reporting/generate_new_figures.py`](experiments/reporting/generate_new_figures.py) | `topk_sparsification_results.json` |
| **Figure 8** | Empirical Attack Defense (MIA & DLG) | [`experiments/reporting/generate_new_figures.py`](experiments/reporting/generate_new_figures.py) | `mia_benchmark_results.json` |

---

## 📂 Repository Directory Layout

```text
.
├── reproduce.py                       # Unified CLI master reproduction runner
├── requirements.txt                   # Pinned Python package dependencies
├── environment.yml                    # Conda environment definition
├── README.md                          # Reproduction dossier and guide
├── LICENSE                            # MIT License
├── data/                              # Benchmark Datasets (~48 MB uncompressed)
│   ├── cart_events.csv                # EARS raw cart interactions
│   ├── clientes.csv                   # EARS client metadata
│   ├── order_items.csv                # EARS order line items
│   ├── orders.csv                     # EARS purchase transactions
│   ├── produtos.csv                   # EARS item catalog & categories
│   ├── viewed_products.csv            # EARS implicit browsing interactions
│   └── movielens/                     # MovieLens-100k benchmark dataset
├── recommender_core/                  # Neural Recommender Architecture
│   ├── Core_Model.py                  # PyTorch hybrid GNN / collaborative model
│   ├── dataset.py                     # Feature graph materializer & twin cache
│   └── train.py                       # Pairwise ranking loss & local SGD
├── support/                           # Signal processing & contracts
│   ├── feature_contract.py            # Feature dimensions & channel contracts
│   ├── signal_utils.py                # Graph embeddings & text extraction
│   └── twin_state.py                  # Customer-twin materialized states
├── api_service/                       # Preprocessing pipeline
│   └── preprocess.py                  # Generic tabular interaction preprocessor
├── experiments/                       # Experimentation Suite
│   ├── config.py                      # Configurations E0 through E4
│   ├── run_experiments.py             # Single-seed experiment runner
│   ├── run_multiseed_benchmark.py     # 10-seed multi-dataset runner
│   ├── attacks/                       # MIA & DLG gradient inversion attacks
│   ├── centralized/                   # Centralized training engine (E0)
│   ├── federated/                     # FedAvg, DP Engine, TenSEAL CKKS HE
│   ├── evaluation/                    # Ranking metrics, statistical power, XAI
│   ├── reporting/                     # Plot and table generators
│   └── results/                       # Empirical multi-seed benchmark logs (JSON)
├── tooling/                           # Manuscript QA tooling
│   └── validate_manuscript.py         # Automated LaTeX citation & label auditor
└── Paper/                             # Complete Manuscript & Artifacts
    ├── main.tex                       # Primary LaTeX document
    ├── tables.tex                     # Publication tables I through XII
    ├── references.bib                 # Bibliography database (Harvard style)
    ├── figures/                       # Rendered PDF & PNG publication figures
    └── main.pdf                       # Fully rendered 45-page manuscript PDF
```

---

## 💻 Hardware & Runtime Specifications

* **Operating System:** Linux (Ubuntu 22.04+) or macOS (macOS 14+ tested).
* **RAM:** Minimum 8 GB RAM (16 GB recommended for full 10-seed benchmarks).
* **CPU/GPU:** Multi-core x86_64 or Apple Silicon ARM64. PyTorch CPU backend is fully supported and deterministic.
* **Estimated Execution Runtimes:**
  - Figure / Table regeneration: $\approx 3$ seconds.
  - Smoke test: $\approx 60$ seconds.
  - Empirical attack benchmark: $\approx 25$ seconds.
  - Top-$K$ sparsification sweep: $\approx 40$ seconds.
  - Full 10-seed multi-dataset evaluation: $\approx 2.5$ hours on 8-core CPU.

---

## 📄 License & Attribution

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
