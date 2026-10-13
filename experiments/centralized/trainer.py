from __future__ import annotations

import time
import tracemalloc
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from experiments.config import ExperimentConfig
from experiments.data.partitioner import ExperimentDataBundle
from experiments.evaluation.performance_metrics import PerformanceTracker
from recommender_core.Core_Model import CoreHybridRecSys
from recommender_core.train import pairwise_rank_loss, _derive_conf_target


def compute_fl_sample_budget(config: ExperimentConfig, data_bundle: ExperimentDataBundle) -> int:
    """
    Computes the exact total number of training samples processed across all FL rounds
    given the deterministic client sampling schedule.
    """
    import random
    rng_state = random.getstate()
    random.seed(config.fl.seed)
    all_clients = list(data_bundle.client_partitions.keys())
    m = min(config.fl.clients_per_round, len(all_clients))
    total_samples = 0
    for _ in range(config.fl.num_rounds):
        sampled = random.sample(all_clients, m)
        for c_idx in sampled:
            total_samples += len(data_bundle.client_partitions[c_idx]) * config.fl.local_epochs
    random.setstate(rng_state)
    return total_samples


def run_centralized_experiment(
    config: ExperimentConfig,
    data_bundle: ExperimentDataBundle,
    device: torch.device,
) -> Tuple[torch.nn.Module, PerformanceTracker]:
    """
    Executes centralized baseline training (E0) on pooled client data with
    exact sample-budget and gradient-step parity matching the federated rounds.
    """
    tracker = PerformanceTracker()
    tracemalloc.start()
    total_start = time.perf_counter()

    model = CoreHybridRecSys(branch_dims=data_bundle.branch_dims)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.fl.learning_rate,
        weight_decay=config.fl.weight_decay,
    )

    dataloader = data_bundle.get_centralized_dataloader(
        batch_size=config.fl.batch_size,
        shuffle=True,
    )

    # Exact sample budget matching the federated rounds
    sample_budget = compute_fl_sample_budget(config, data_bundle)
    total_expected_batches = (sample_budget + config.fl.batch_size - 1) // config.fl.batch_size
    print(f"  [E0 Training] Target budget: {sample_budget:,} interaction samples (~{total_expected_batches} gradient steps)")

    processed_samples = 0
    step = 0
    epoch_idx = 0
    log_interval = max(10, total_expected_batches // 10)

    done = False
    while not done:
        epoch_idx += 1
        dataloader = data_bundle.get_centralized_dataloader(
            batch_size=config.fl.batch_size,
            shuffle=True,
        )
        for batch in dataloader:
            inputs = {
                "user_state": {k: v.to(device) for k, v in batch["user_state"].items()},
                "collaborative": {k: v.to(device) for k, v in batch["collaborative"].items()},
                "pair_context": {k: v.to(device) for k, v in batch["pair_context"].items()},
            }
            labels_reg = batch["label_reg"].float().to(device)
            labels_rank = batch["label_rank"].float().to(device)
            user_idx = batch["user_idx"].long().to(device)

            optimizer.zero_grad()
            purchase_logit, rank_score, conf_logit = model(inputs)

            loss_bce = F.binary_cross_entropy_with_logits(purchase_logit, labels_reg)
            loss_rank = pairwise_rank_loss(rank_score, labels_rank, user_idx)
            conf_target = _derive_conf_target(purchase_logit, rank_score, labels_reg, labels_rank)
            loss_conf = F.binary_cross_entropy_with_logits(conf_logit, conf_target)

            loss = loss_bce + 0.2 * loss_rank + 0.1 * loss_conf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            b_size = len(user_idx)
            processed_samples += b_size
            step += 1

            if step % log_interval == 0 or processed_samples >= sample_budget:
                pct = min(100.0, (processed_samples / sample_budget) * 100.0)
                elapsed = time.perf_counter() - total_start
                print(f"  [E0 Training] Step {step}/{total_expected_batches} ({pct:5.1f}%) | "
                      f"Samples: {processed_samples:,}/{sample_budget:,} | "
                      f"Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s")

            if processed_samples >= sample_budget:
                done = True
                break

    total_time = time.perf_counter() - total_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracker.total_training_time_sec = total_time
    tracker.local_training_time_sec = total_time
    tracker.total_execution_time_sec = total_time
    tracker.peak_memory_mb = peak / (1024.0 * 1024.0)

    return model, tracker
