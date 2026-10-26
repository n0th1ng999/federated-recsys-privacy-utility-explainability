from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from experiments.config import ExperimentConfig
from experiments.federated.dp_engine import DPEngine
from experiments.federated.he_engine import HEEngine
from experiments.evaluation.performance_metrics import PerformanceTracker
from recommender_core.Core_Model import CoreHybridRecSys


class FederatedServer:
    """
    Aggregation server for E1-E4.

    Aggregation rules (as stated in Algorithm 1 of the manuscript):
      * E1, E3: sample-weighted FedAvg, w_k = n_k / sum_j n_j.
      * E2, E4: uniform mean over the m sampled clients (DP-FedAvg with fixed
        denominator), so that the replace-one sensitivity of the mean is exactly 2C/m.

    Where noise is added:
      * E2 (central DP): the server receives each client's plaintext update, clips
        it and adds noise to the mean. The server is therefore trusted; the DP
        guarantee covers what is released AFTER aggregation (the global models).
      * E4 (distributed DP under CKKS secure aggregation): clients clip and add
        their share of the noise before encrypting (see fl_runner); this class
        only sees ciphertexts.

    `last_observation` keeps what each party saw in the most recent round, for the
    reconstruction attacks.
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

        self.dp_engine: Optional[DPEngine] = None
        if config.dp.enabled:
            self.dp_engine = DPEngine(
                clipping_norm=config.dp.clipping_norm,
                noise_multiplier=config.dp.noise_multiplier,
                target_delta=config.dp.target_delta,
            )

        self.he_engine: Optional[HEEngine] = None
        if config.he.enabled:
            self.he_engine = HEEngine(
                poly_modulus_degree=config.he.poly_modulus_degree,
                coeff_mod_bit_sizes=config.he.coeff_mod_bit_sizes,
                scale_bits=config.he.scale_bits,
            )

        self.aggregation = "uniform" if config.dp.enabled else "sample_weighted"
        self.last_observation: Dict[str, Any] = {}
        self._clip_events: List[bool] = []
        self._preclip_norms: List[float] = []

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().detach().clone() for k, v in self.global_model.state_dict().items()}

    def aggregation_weights(self, sample_counts: List[int]) -> List[float]:
        m = len(sample_counts)
        if self.aggregation == "uniform":
            return [1.0 / m] * m
        total = max(1, sum(sample_counts))
        return [cnt / total for cnt in sample_counts]

    def record_clip(self, preclip_norm: float) -> None:
        self._preclip_norms.append(preclip_norm)
        self._clip_events.append(preclip_norm > self.config.dp.clipping_norm)

    def clip_stats(self) -> Dict[str, Optional[float]]:
        if not self._preclip_norms:
            return {"clip_fraction": None, "mean_preclip_norm": None}
        return {
            "clip_fraction": float(np.mean(self._clip_events)),
            "mean_preclip_norm": float(np.mean(self._preclip_norms)),
        }

    @staticmethod
    def _weighted_sum(updates: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        out = {}
        for key in updates[0]:
            out[key] = torch.stack([u[key] * w for u, w in zip(updates, weights)], dim=0).sum(dim=0)
        return out

    def aggregate_plaintext(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
        tracker: PerformanceTracker,
    ) -> Dict[str, torch.Tensor]:
        """E1 (weighted FedAvg) and E2 (central DP, uniform mean + server-side noise)."""
        agg_t0 = time.perf_counter()
        weights = self.aggregation_weights(sample_counts)
        m = len(client_updates)

        if self.config.dp.enabled:
            # The server sees these plaintext, un-noised updates (trusted curator).
            self.last_observation = {"server_sees": "plaintext_client_updates", "client_updates": client_updates}
            clipped_updates = []
            for upd in client_updates:
                clipped, norm = self.dp_engine.clip_update(upd)
                self.record_clip(norm)
                clipped_updates.append(clipped)
            mean_update = self._weighted_sum(clipped_updates, weights)
            noisy = self.dp_engine.add_gaussian_noise(mean_update, self.dp_engine.central_noise_std(m))
            self.dp_engine.step_accounting(self.config.fl.num_clients, m)
            self.last_observation["released_aggregate"] = noisy
            tracker.aggregation_time_sec += time.perf_counter() - agg_t0
            return noisy

        self.last_observation = {"server_sees": "plaintext_client_updates", "client_updates": client_updates}
        avg = self._weighted_sum(client_updates, weights)
        self.last_observation["released_aggregate"] = avg
        tracker.aggregation_time_sec += time.perf_counter() - agg_t0
        return avg

    def aggregate_encrypted(
        self,
        client_payloads: List[List[bytes]],
        sample_counts: List[int],
        packed_len: int,
        tracker: PerformanceTracker,
    ) -> np.ndarray:
        """
        E3 and E4 (and shared-mask Top-K): homomorphic aggregation without the secret key,
        then key-holder decryption. Returns the decrypted packed vector (length packed_len).
        """
        agg_t0 = time.perf_counter()
        weights = self.aggregation_weights(sample_counts)

        enc_agg, comp_t, agg_bytes = self.he_engine.server_aggregate(client_payloads, weights)
        tracker.he_computation_time_sec += comp_t

        dec_flat, dec_t = self.he_engine.key_holder_decrypt(enc_agg, packed_len)
        tracker.he_decryption_time_sec += dec_t

        if self.config.dp.enabled:
            self.dp_engine.step_accounting(self.config.fl.num_clients, len(client_payloads))

        self.last_observation = {
            "server_sees": "ciphertexts_only",
            "ciphertext_bytes_per_client": [sum(len(b) for b in p) for p in client_payloads],
            "key_holder_sees": "decrypted_aggregate",
        }
        tracker.aggregation_time_sec += time.perf_counter() - agg_t0
        return dec_flat

    def aggregate_sparse_plaintext(
        self,
        client_sparse: List[tuple],
        sample_counts: List[int],
        full_len: int,
        tracker: PerformanceTracker,
    ) -> np.ndarray:
        """Per-client Top-K in plaintext: weighted scatter-add of (indices, values)."""
        agg_t0 = time.perf_counter()
        weights = self.aggregation_weights(sample_counts)
        out = np.zeros(full_len, dtype=np.float64)
        for (idx, vals), w in zip(client_sparse, weights):
            np.add.at(out, idx, w * vals)
        tracker.aggregation_time_sec += time.perf_counter() - agg_t0
        return out

    def apply_update(self, global_update: Dict[str, torch.Tensor]) -> None:
        """Applies aggregated update to the global model: W = W + Delta."""
        current_state = self.global_model.state_dict()
        new_state = {}
        for key in current_state:
            new_state[key] = current_state[key] + global_update[key].to(self.device)
        self.global_model.load_state_dict(new_state)
