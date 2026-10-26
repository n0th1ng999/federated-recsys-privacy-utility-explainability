# Experiments

| Path | Purpose |
|---|---|
| `config.py` | Configurations E0, E1, E2 (σ = 1, 2, 4, 8), E3, E4 (σ = 4) and the Top-k variants |
| `run_experiment.py` | One dataset × one seed: train every configuration, evaluate on the temporal hold-out, run membership inference and the explanation protocol, write `results/main/<dataset>/<config>/seed_<n>.json` |
| `run_sweep.py` | All datasets and seeds in parallel; `--timing` runs the sequential timing pass into `results/timing` |
| `attacks/membership.py`, `attacks/reconstruction.py` | Membership inference by privacy unit; gradient inversion by observation scenario |
| `federated/` | FL loop, DP engine and accountant, CKKS secure aggregation, shared-mask Top-k |
| `evaluation/xai_protocol.py` | Explainer budgets, removal curves, agreement with the centralised model |
| `reporting/export_latex.py` | Generates every table and value of the manuscript into `Paper/generated/` |
| `tests/` | Checks of the accountant, noise calibration, CKKS aggregation, split and XAI wrapper |

See the repository README for the full replication procedure.
