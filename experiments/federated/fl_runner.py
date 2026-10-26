from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import dp_accounting
import numpy as np
import torch
from dp_accounting import rdp

from experiments.config import ExperimentConfig
from experiments.data.partitioner import ExperimentDataBundle
from experiments.evaluation.performance_metrics import PerformanceTracker
from experiments.federated.client import FederatedClient
from experiments.federated.dp_engine import RDP_ORDERS
from experiments.federated.he_engine import HEEngine
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
    if config.dp.enabled and m != config.fl.clients_per_round:
        raise ValueError("DP accounting assumes exactly clients_per_round clients are sampled each round.")

    # Plaintext model payload (float32 state dict) for broadcasts and plaintext uplinks
    sample_weights = server.get_global_weights()
    standard_payload_bytes = sum(t.element_size() * t.nelement() for t in sample_weights.values())

    tracker.aggregation = server.aggregation
    if config.he.enabled:
        tracker.he_keygen_time_sec = server.he_engine.keygen_time_sec
        # Public CKKS context (no secret key) is distributed once to every client.
        tracker.downlink_he_context_bytes = server.he_engine.public_context_bytes * len(all_client_indices)

    participation = {c: 0 for c in all_client_indices}

    topk_mode = config.fl.topk_mode
    if topk_mode == "shared_mask" and not config.he.enabled:
        raise ValueError("shared_mask Top-K is defined for the CKKS configuration")
    if topk_mode == "per_client" and config.he.enabled:
        raise ValueError("per-client Top-K indices differ across clients and cannot be summed under CKKS")
    _, model_shapes = HEEngine.flatten_weights(sample_weights)
    full_len = sum(length for _, _, length in model_shapes)
    k_keep = max(1, int(round(config.fl.topk_ratio * full_len))) if topk_mode != "none" else full_len
    mask_rng = np.random.default_rng(config.fl.seed + 7919)
    shared_mask = None
    last_global_flat = None
    tracker.topk_mode = topk_mode
    tracker.topk_ratio = config.fl.topk_ratio if topk_mode != "none" else 1.0
    tracker.topk_coordinates = k_keep

    print(f"  Starting Federated Training: {config.fl.num_rounds} rounds, {len(all_client_indices)} total clients, {m} clients sampled/round")

    for round_idx in range(1, config.fl.num_rounds + 1):
        round_t0 = time.perf_counter()
        # Fixed-size sampling without replacement (the scheme the DP accountant assumes)
        sampled_indices = random.sample(all_client_indices, m)
        for c in sampled_indices:
            participation[c] += 1
        print(f"  [FL Round {round_idx:02d}/{config.fl.num_rounds:02d}] Sampled {m} clients: {sampled_indices}")

        global_weights = server.get_global_weights()
        tracker.downlink_model_bytes += standard_payload_bytes * len(sampled_indices)
        tracker.communication_bytes_received += standard_payload_bytes * len(sampled_indices)

        if topk_mode == "shared_mask":
            # Server picks the coordinates every client will send this round.
            if last_global_flat is None:
                shared_mask = np.sort(mask_rng.choice(full_len, size=k_keep, replace=False))
            else:
                shared_mask = np.sort(np.argpartition(-np.abs(last_global_flat), k_keep - 1)[:k_keep])
            # Broadcast as a bitmap or as int32 indices, whichever is smaller.
            mask_bytes = min(int(math.ceil(full_len / 8)), 4 * k_keep)
            tracker.downlink_mask_bytes += mask_bytes * len(sampled_indices)
            tracker.communication_bytes_received += mask_bytes * len(sampled_indices)

        client_updates = []
        client_payloads = []
        client_sparse = []
        sample_counts = []
        shapes = None
        packed_len = None

        local_t0 = time.perf_counter()
        for c_idx in sampled_indices:
            client = clients[c_idx]
            delta, n_samples, local_t, peak_mb = client.train_epoch(
                global_weights=global_weights,
                local_epochs=config.fl.local_epochs,
            )
            sample_counts.append(n_samples)
            tracker.local_training_time_sec += local_t
            tracker.peak_memory_mb = max(tracker.peak_memory_mb, peak_mb)

            if config.he.enabled:
                if config.dp.enabled:
                    # E4, client side: clip, add this client's share of the noise, then encrypt.
                    clipped, norm = server.dp_engine.clip_update(delta)
                    server.record_clip(norm)
                    delta = server.dp_engine.add_gaussian_noise(clipped, server.dp_engine.distributed_noise_std(m))
                flat, shapes = server.he_engine.flatten_weights(delta)
                t_sp = time.perf_counter()
                if shared_mask is not None:
                    flat = flat[shared_mask]
                tracker.sparsification_time_sec += time.perf_counter() - t_sp
                packed_len = len(flat)
                payload, enc_t, ct_bytes = server.he_engine.client_encrypt(flat)
                client_payloads.append(payload)
                tracker.he_encryption_time_sec += enc_t
                tracker.he_ciphertext_size_bytes += ct_bytes
                tracker.uplink_bytes += ct_bytes
                tracker.communication_bytes_sent += ct_bytes
            elif topk_mode == "per_client":
                flat, shapes = HEEngine.flatten_weights(delta)
                t_sp = time.perf_counter()
                idx = np.argpartition(-np.abs(flat), k_keep - 1)[:k_keep].astype(np.int32)
                vals = flat[idx].astype(np.float32)
                tracker.sparsification_time_sec += time.perf_counter() - t_sp
                client_sparse.append((idx, vals.astype(np.float64)))
                sparse_bytes = idx.nbytes + vals.nbytes  # int32 index + float32 value per coordinate
                tracker.uplink_bytes += sparse_bytes
                tracker.communication_bytes_sent += sparse_bytes
            else:
                client_updates.append(delta)
                tracker.uplink_bytes += standard_payload_bytes
                tracker.communication_bytes_sent += standard_payload_bytes

        local_elapsed = time.perf_counter() - local_t0
        print(f"    -> Local training: {sum(sample_counts):,} samples processed across {m} clients in {local_elapsed:.2f}s")

        agg_t0 = time.perf_counter()
        if config.he.enabled:
            packed = server.aggregate_encrypted(client_payloads, sample_counts, packed_len, tracker)
            if shared_mask is not None:
                full = np.zeros(full_len, dtype=np.float64)
                full[shared_mask] = packed
            else:
                full = packed
            global_update = server.he_engine.unflatten_weights(full, shapes)
        elif topk_mode == "per_client":
            full = server.aggregate_sparse_plaintext(client_sparse, sample_counts, full_len, tracker)
            global_update = HEEngine.unflatten_weights(full, shapes)
        else:
            global_update = server.aggregate_plaintext(client_updates, sample_counts, tracker)
            full = None
        agg_dur = time.perf_counter() - agg_t0
        if topk_mode == "shared_mask":
            last_global_flat = full

        if config.dp.enabled and round_idx == config.fl.num_rounds:
            # Accounting is evaluated once at the end: the RDP computation for sampling without
            # replacement is expensive and would otherwise inflate wall-clock training time.
            eps_now, _ = server.dp_engine.get_privacy_spent()
            print(f"    -> DP ({tracker.aggregation} mean, C={config.dp.clipping_norm}, sigma={config.dp.noise_multiplier}) | eps <= {eps_now:.3f} after {round_idx} rounds")
        if config.he.enabled:
            print(f"    -> CKKS secure aggregation ({agg_dur:.2f}s, uplink so far: {tracker.uplink_bytes/(1024*1024):.1f} MB)")

        server.apply_update(global_update)
        round_dur = time.perf_counter() - round_t0
        print(f"  [FL Round {round_idx:02d}/{config.fl.num_rounds:02d}] Finished in {round_dur:.2f}s | Elapsed: {time.perf_counter() - total_start:.1f}s")

    tracker.total_training_time_sec = time.perf_counter() - total_start
    tracker.total_execution_time_sec = tracker.total_training_time_sec
    tracker.client_participation = dict(participation)

    if config.dp.enabled:
        eps, delta = server.dp_engine.get_privacy_spent()
        tracker.dp_epsilon = round(eps, 4)
        tracker.dp_delta = delta
        tracker.dp_clipping_norm = config.dp.clipping_norm
        tracker.dp_noise_multiplier = config.dp.noise_multiplier
        tracker.dp_accountant = "dp_accounting RDP, sampling w/o replacement, replace-one, client-level"
        stats = server.clip_stats()
        tracker.dp_clip_fraction = stats["clip_fraction"]
        tracker.dp_mean_preclip_norm = stats["mean_preclip_norm"]
        if config.he.enabled:
            tracker.dp_noise_location = "client_distributed"
            tracker.dp_epsilon_single_update_local = round(local_epsilon_single_client(
                noise_multiplier=config.dp.noise_multiplier,
                clients_per_round=m,
                participations=max(participation.values()),
                delta=config.dp.target_delta,
            ), 4)
        else:
            tracker.dp_noise_location = "server"

    return server.global_model, tracker


def local_epsilon_single_client(noise_multiplier: float, clients_per_round: int, participations: int, delta: float) -> float:
    """
    Epsilon of ONE client's noisy updates if an observer could see them individually
    (i.e. if CKKS confidentiality or key management failed). Each update carries
    noise std sigma*C/sqrt(m) against replace-one sensitivity 2C, i.e. noise multiplier
    sigma/(2*sqrt(m)), composed over that client's participations with no
    subsampling amplification (the observer knows who participated).
    """
    z = noise_multiplier / (2.0 * math.sqrt(clients_per_round))
    acc = rdp.RdpAccountant(orders=RDP_ORDERS, neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE)
    acc.compose(dp_accounting.SelfComposedDpEvent(dp_accounting.GaussianDpEvent(z), participations))
    return float(acc.get_epsilon(delta))
