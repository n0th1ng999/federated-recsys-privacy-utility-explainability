from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

import torch

from experiments.config import ExperimentConfig
from experiments.data.partitioner import ExperimentDataBundle
from experiments.evaluation.performance_metrics import PerformanceTracker
from experiments.federated.client import FederatedClient
from experiments.federated.server import FederatedServer


def run_federated_experiment(
    config: ExperimentConfig,
    data_bundle: ExperimentDataBundle,
    device: torch.device,
) -> Tuple[torch.nn.Module, PerformanceTracker]:
    """
    Executes a complete federated training simulation across rounds for E1, E2, E3, or E4.
    """
    random.seed(config.fl.seed)
    torch.manual_seed(config.fl.seed)

    tracker = PerformanceTracker()
    total_start = time.perf_counter()

    server = FederatedServer(
        branch_dims=data_bundle.branch_dims,
        config=config,
        device=device,
    )

    # Initialize client runners
    clients: Dict[int, FederatedClient] = {}
    for client_idx in data_bundle.client_partitions:
        dl = data_bundle.get_client_dataloader(
            client_idx=client_idx,
            batch_size=config.fl.batch_size,
            shuffle=True,
        )
        clients[client_idx] = FederatedClient(
            client_id=client_idx,
            dataloader=dl,
            branch_dims=data_bundle.branch_dims,
            device=device,
            learning_rate=config.fl.learning_rate,
            weight_decay=config.fl.weight_decay,
        )

    all_client_indices = list(clients.keys())
    m = min(config.fl.clients_per_round, len(all_client_indices))

    # Weight size in bytes for standard communication tracking
    sample_weights = server.get_global_weights()
    standard_payload_bytes = sum(t.element_size() * t.nelement() for t in sample_weights.values())

    print(f"  Starting Federated Training: {config.fl.num_rounds} rounds, {len(all_client_indices)} total clients, {m} clients sampled/round")

    for round_idx in range(1, config.fl.num_rounds + 1):
        round_t0 = time.perf_counter()
        sampled_indices = random.sample(all_client_indices, m)
        print(f"  [FL Round {round_idx:02d}/{config.fl.num_rounds:02d}] Sampled {m} clients: {sampled_indices}")

        global_weights = server.get_global_weights()
        # Broadcast downlink communication
        tracker.communication_bytes_received += (standard_payload_bytes * len(sampled_indices))

        client_updates = []
        sample_counts = []

        local_t0 = time.perf_counter()
        for c_idx in sampled_indices:
            client = clients[c_idx]
            delta, n_samples, local_t, peak_mb = client.train_epoch(
                global_weights=global_weights,
                local_epochs=config.fl.local_epochs,
            )
            client_updates.append(delta)
            sample_counts.append(n_samples)

            tracker.local_training_time_sec += local_t
            tracker.peak_memory_mb = max(tracker.peak_memory_mb, peak_mb)
            # Uplink communication
            tracker.communication_bytes_sent += standard_payload_bytes

        local_elapsed = time.perf_counter() - local_t0
        tot_samples = sum(sample_counts)
        print(f"    -> Local training: {tot_samples:,} samples processed across {m} clients in {local_elapsed:.2f}s")

        # Server aggregates updates
        agg_t0 = time.perf_counter()
        global_update = server.aggregate_updates(
            client_updates=client_updates,
            sample_counts=sample_counts,
            tracker=tracker,
        )
        agg_dur = time.perf_counter() - agg_t0

        if config.dp.enabled and server.dp_engine is not None:
            eps_now, _ = server.dp_engine.get_privacy_spent()
            print(f"    -> Server DP: Clipped (C={config.dp.clipping_norm}) & Gaussian noise injected (σ={config.dp.noise_multiplier}) | Cumulative ε={eps_now:.3f}")
        if config.he.enabled:
            print(f"    -> Server HE: CKKS encrypted aggregation & decryption ({agg_dur:.2f}s, ciphertext: {tracker.he_ciphertext_size_bytes/(1024*1024):.1f} MB)")
        elif not config.dp.enabled:
            print(f"    -> Server: Plaintext FedAvg aggregation completed in {agg_dur:.4f}s")

        # Server applies update
        server.apply_update(global_update)
        round_dur = time.perf_counter() - round_t0
        print(f"  [FL Round {round_idx:02d}/{config.fl.num_rounds:02d}] Finished in {round_dur:.2f}s | Elapsed: {time.perf_counter() - total_start:.1f}s")

    tracker.total_training_time_sec = time.perf_counter() - total_start
    tracker.total_execution_time_sec = tracker.total_training_time_sec

    # Log DP telemetry if enabled
    if config.dp.enabled and server.dp_engine is not None:
        eps, delta = server.dp_engine.get_privacy_spent()
        tracker.dp_epsilon = round(eps, 4)
        tracker.dp_delta = delta
        tracker.dp_clipping_norm = config.dp.clipping_norm
        tracker.dp_noise_multiplier = config.dp.noise_multiplier

    return server.global_model, tracker
