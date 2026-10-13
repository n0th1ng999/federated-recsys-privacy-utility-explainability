from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from experiments.config import ExperimentConfig
from experiments.federated.dp_engine import DPEngine
from experiments.federated.he_engine import HEEngine
from experiments.evaluation.performance_metrics import PerformanceTracker
from recommender_core.Core_Model import CoreHybridRecSys


class FederatedServer:
    """
    Central orchestrator for Federated Learning rounds. Handles client update aggregation
    using standard FedAvg, Differential Privacy, Homomorphic Encryption, or their combination.
    """

    def __init__(
        self,
        branch_dims: Dict[str, Any],
        config: ExperimentConfig,
        device: torch.device,
    ):
        self.branch_dims = branch_dims
        self.config = config
        self.device = device

        self.global_model = CoreHybridRecSys(branch_dims=branch_dims)
        self.global_model.to(self.device)

        # Differential privacy engine
        self.dp_engine: Optional[DPEngine] = None
        if config.dp.enabled:
            self.dp_engine = DPEngine(
                clipping_norm=config.dp.clipping_norm,
                noise_multiplier=config.dp.noise_multiplier,
                target_delta=config.dp.target_delta,
            )

        # Homomorphic encryption engine
        self.he_engine: Optional[HEEngine] = None
        if config.he.enabled:
            self.he_engine = HEEngine(
                poly_modulus_degree=config.he.poly_modulus_degree,
                coeff_mod_bit_sizes=config.he.coeff_mod_bit_sizes,
                scale_bits=config.he.scale_bits,
            )

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().detach().clone() for k, v in self.global_model.state_dict().items()}

    def aggregate_updates(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        tracker: PerformanceTracker,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates client weight deltas into a global update according to the active configuration.
        """
        agg_t0 = time.perf_counter()
        num_clients = len(client_updates)
        total_samples = sum(sample_counts)
        weights = [cnt / max(1, total_samples) for cnt in sample_counts]

        # -------------------------------------------------------------
        # Branch 1: Differential Privacy (E2)
        # -------------------------------------------------------------
        if self.config.dp.enabled and not self.config.he.enabled:
            clipped_updates = []
            for upd in client_updates:
                clipped, _ = self.dp_engine.clip_update(upd)
                clipped_updates.append(clipped)

            # Weighted average of clipped updates
            avg_update = {}
            for key in clipped_updates[0]:
                stacked = torch.stack([u[key] * w for u, w in zip(clipped_updates, weights)], dim=0)
                avg_update[key] = stacked.sum(dim=0)

            # Add calibrated Gaussian noise
            noisy_update = self.dp_engine.add_noise_to_aggregated_update(
                avg_update,
                num_sampled_clients=num_clients,
            )
            q = num_clients / max(1, self.config.fl.num_clients)
            self.dp_engine.step_accounting(q=q)

            tracker.aggregation_time_sec += (time.perf_counter() - agg_t0)
            return noisy_update

        # -------------------------------------------------------------
        # Branch 2: Homomorphic Encryption (E3)
        # -------------------------------------------------------------
        if self.config.he.enabled and not self.config.dp.enabled:
            tracker.he_keygen_time_sec = self.he_engine.keygen_time_sec
            encrypted_updates = []
            shapes = None

            for upd in client_updates:
                flat, shapes = self.he_engine.flatten_weights(upd)
                enc_vec, enc_t, ct_size = self.he_engine.encrypt_vector(flat)
                encrypted_updates.append(enc_vec)
                tracker.he_encryption_time_sec += enc_t
                tracker.he_ciphertext_size_bytes += ct_size
                tracker.communication_bytes_sent += ct_size

            # Homomorphic addition in ciphertext domain
            enc_aggregate, comp_t = self.he_engine.homomorphic_aggregate(encrypted_updates, weights)
            tracker.he_computation_time_sec += comp_t

            # Decryption
            dec_flat, dec_t = self.he_engine.decrypt_vector(enc_aggregate)
            tracker.he_decryption_time_sec += dec_t

            final_update = self.he_engine.unflatten_weights(dec_flat, shapes, device=torch.device("cpu"))
            tracker.aggregation_time_sec += (time.perf_counter() - agg_t0)
            return final_update

        # -------------------------------------------------------------
        # Branch 3: Combined DP + HE (E4)
        # -------------------------------------------------------------
        if self.config.he.enabled and self.config.dp.enabled:
            tracker.he_keygen_time_sec = self.he_engine.keygen_time_sec
            encrypted_updates = []
            shapes = None

            for upd in client_updates:
                # Client-side clipping and DP noise injection before encryption
                clipped, _ = self.dp_engine.clip_update(upd)
                local_noisy = self.dp_engine.add_noise_to_aggregated_update(
                    clipped,
                    num_sampled_clients=num_clients,
                )
                flat, shapes = self.he_engine.flatten_weights(local_noisy)
                enc_vec, enc_t, ct_size = self.he_engine.encrypt_vector(flat)
                encrypted_updates.append(enc_vec)
                tracker.he_encryption_time_sec += enc_t
                tracker.he_ciphertext_size_bytes += ct_size
                tracker.communication_bytes_sent += ct_size

            # Homomorphic aggregation
            enc_aggregate, comp_t = self.he_engine.homomorphic_aggregate(encrypted_updates, weights)
            tracker.he_computation_time_sec += comp_t

            # Decryption
            dec_flat, dec_t = self.he_engine.decrypt_vector(enc_aggregate)
            tracker.he_decryption_time_sec += dec_t

            q = num_clients / max(1, self.config.fl.num_clients)
            self.dp_engine.step_accounting(q=q)

            final_update = self.he_engine.unflatten_weights(dec_flat, shapes, device=torch.device("cpu"))
            tracker.aggregation_time_sec += (time.perf_counter() - agg_t0)
            return final_update

        # -------------------------------------------------------------
        # Branch 4: Standard FedAvg (E1 Baseline)
        # -------------------------------------------------------------
        avg_update = {}
        for key in client_updates[0]:
            stacked = torch.stack([u[key] * w for u, w in zip(client_updates, weights)], dim=0)
            avg_update[key] = stacked.sum(dim=0)

        tracker.aggregation_time_sec += (time.perf_counter() - agg_t0)
        return avg_update

    def apply_update(self, global_update: Dict[str, torch.Tensor]) -> None:
        """Applies aggregated update to the global model: W = W + Delta."""
        current_state = self.global_model.state_dict()
        new_state = {}
        for key in current_state:
            new_state[key] = current_state[key] + global_update[key].to(self.device)
        self.global_model.load_state_dict(new_state)
