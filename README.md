# Replication package

Privacy, Utility and Explainability Trade-offs in Privacy-Preserving Federated Recommender Systems (anonymised for review).

This package contains the code, configurations, data and per-run results behind every number in the manuscript. It also contains the scripts that turn those results into the manuscript's tables and text values.

## Contents

| Path | What it is |
|---|---|
| `experiments/config.py` | All configurations: E0, E1, E2 (σ = 1, 2, 4, 8), E3, E4 (σ = 4), Top-k variants |
| `experiments/data/partitioner.py` | Temporal hold-out split, user groups, record hold-out for membership inference |
| `experiments/data/movielens_adapter.py` | Downloads MovieLens-100k from GroupLens and converts it |
| `experiments/federated/` | FL loop, central and distributed DP (`dp_engine.py`), CKKS secure aggregation (`he_engine.py`), shared-mask Top-k |
| `experiments/attacks/membership.py` | Membership inference (client participation and record membership) |
| `experiments/attacks/reconstruction.py` | Gradient inversion on the recommender, by observation scenario |
| `experiments/run_controls.py` | Untrained-network and noise-only controls (initial model plus the exact E2 noise, 15 rounds) |
| `experiments/evaluation/intervention_plausibility.py` | Nearest-neighbour distance of marginal and conditional replacements to real data |
| `experiments/evaluation/xai_protocol.py` | KernelSHAP/LIME with measured budgets, removal curves, agreement with E0 |
| `experiments/run_experiment.py` | One dataset × one seed: train, evaluate, attack, explain; writes one JSON per configuration |
| `experiments/run_sweep.py` | Runs all datasets and seeds; `--timing` runs the sequential timing pass |
| `experiments/reporting/export_latex.py` | Builds every table and text value of the paper from the result files |
| `experiments/tests/` | Checks of the accountant, noise calibration, CKKS aggregation, split disjointness and the XAI wrapper |
| `experiments/results/main/<dataset>/<config>/seed_<n>.json` | Per-run results with provenance (code hash, library versions, split) |
| `experiments/results/main/attacks/reconstruction_seed_<n>.json` | Gradient-inversion results per seed, including per-record values |
| `experiments/results/timing/` | Sequential timing runs used for the overhead and cost tables |
| `experiments/results/main/summary.json` | Machine-readable summary of every reported number |
| `Paper/generated/` | Generated tables (`tab_*.tex`), text values (`results_macros.tex`), Figure S1 |
| `data/` | Synthetic EARS dataset (no real personal data) |

## Environment

```bash
conda env create -f environment.yml
conda activate recsys-replication
```

TenSEAL is required. The CKKS experiments refuse to run without it; there is no simulated fallback.

## Reproducing the results

```bash
python -m pytest experiments/tests                                  # sanity checks (about 1 minute)
python experiments/run_sweep.py --workers 6                         # utility, MIA, explanations, Top-k: 2 datasets x 10 seeds
python experiments/run_controls.py                                  # untrained-network and noise-only controls
python experiments/evaluation/intervention_plausibility.py          # plausibility of marginal vs conditional replacement (EARS)
python experiments/attacks/reconstruction.py                        # gradient inversion, 10 seeds
python experiments/run_sweep.py --timing --workers 1 --datasets ears  # timing and volumes, sequential
python experiments/reporting/export_latex.py                        # tables and values in Paper/generated
sh Paper/build.sh                                                   # supplement first, then main (needs tectonic)
```

`run_controls.py` and `intervention_plausibility.py` read the E0 SHAP attributions and splits written by the sweep, so run them after it.

A single run looks like this:

```bash
python experiments/run_experiment.py --dataset ears --seed 42 --configs E1 E3
```

On a 12-core laptop CPU (no GPU used), the sweep takes about 1.5 hours with 6 workers. The reconstruction attack takes about 2.5 hours with 2 processes, and the timing pass about 2 hours. CKKS encryption uses fresh randomness, so E3 and E4 runs are not bit-identical across repetitions. Their differences are within CKKS approximation error.

## Which file produces which table

| Manuscript | Generated file | Source results |
|---|---|---|
| Table I (guarantees) | `tab_guarantees.tex` | `main/ears/E2_*`, `main/ears/E4` |
| Table II (utility) | `tab_utility.tex` | `main/{ears,movielens}/*`, `baselines/` |
| Table III (overhead) | `tab_overhead.tex` | `timing/ears/*` |
| Table IV (Top-k) | `tab_topk.tex` | `main/ears/E*_topk*`, `timing/ears/E*_topk*` |
| Table V (membership inference) | `tab_mia.tex` | `membership_inference` field of `main/ears/*` |
| Table VI (gradient inversion) | `tab_dlg.tex` | `main/attacks/reconstruction_seed_*.json` |
| Table VII (explanations) | `tab_xai.tex` | `explainability` field of `main/ears/*` |
| Tables S3–S10, Figure S1 | `tab_seeds`, `tab_dp`, `tab_ml_detail`, `tab_xai_budget`, `tab_cost`, `tab_mcda`, `fig_removal_curves` | as above |

## Design notes

- **Privacy unit.** One client is one user, so client-level DP is user-level DP.
- **Accounting.** Accounting uses `dp_accounting`'s RDP accountant for Gaussian noise with sampling without replacement. The neighbouring relation is replace-one, and the sensitivity of the uniform mean is 2C/m. Values are upper bounds.
- **E4.** Each client adds noise with standard deviation σC/√m before encrypting. The guarantee on the released models assumes secure aggregation holds (see manuscript Section 3.3).
- **Temporal split.** Nothing dated after the cutoff enters features, statistics or training.

## Provenance of the result files

Each run file stores `provenance.code_sha256`, a hash over the Python files in `experiments/` (excluding results), `recommender_core/`, `support/` and `api_service/preprocess.py`. The released trees carry two hashes:

| Tree | Hash prefix | Differences from the released code |
|---|---|---|
| `results/main/` | `798f606e92bc` | The public CKKS context still included Galois keys, which aggregation never uses. Context bytes in these files are therefore about 35 MB per client. Per-round ε was recomputed at every round instead of once at the end, which gives the same final value. |
| `results/timing/` | `7f551e2e8096` | None in training, DP, CKKS or evaluation code. |

Training, DP noise, aggregation and evaluation are numerically identical across the two code states. Only the serialised context size and wall-clock time differ. For this reason, every byte count and timing in the paper (Tables III and IV, Table S8) comes from `timing`, and `main` supplies utility, attacks and explanations. The released code hashes to `37435af0ec38`. Its training, DP, CKKS and evaluation code is the same as in the `timing` state. The rest differs in three ways, all introduced after both runs:

- the reporting, control and plausibility scripts were added;
- the entry script was renamed to `run_experiment.py` and the result folders to `main/` and `timing/`;
- an empty `experiments/tests/__init__.py` was added. The control (`main/<dataset>/controls/`), plausibility (`main/ears/plausibility/`) and baseline files do not store a code hash.
