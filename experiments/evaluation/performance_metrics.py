from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PerformanceTracker:
    total_training_time_sec: float = 0.0
    local_training_time_sec: float = 0.0
    aggregation_time_sec: float = 0.0
    communication_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    total_execution_time_sec: float = 0.0
    peak_memory_mb: float = 0.0
    communication_bytes_sent: int = 0
    communication_bytes_received: int = 0

    # HE-specific metrics
    he_keygen_time_sec: float = 0.0
    he_encryption_time_sec: float = 0.0
    he_computation_time_sec: float = 0.0
    he_decryption_time_sec: float = 0.0
    he_ciphertext_size_bytes: int = 0

    # DP-specific metrics
    dp_epsilon: Optional[float] = None
    dp_delta: Optional[float] = None
    dp_clipping_norm: Optional[float] = None
    dp_noise_multiplier: Optional[float] = None
    dp_noise_location: Optional[str] = None  # "server" (E2) | "client_distributed" (E4)
    dp_epsilon_single_update_local: Optional[float] = None  # E4: one client's noisy update viewed alone
    dp_accountant: Optional[str] = None
    dp_clip_fraction: Optional[float] = None
    dp_mean_preclip_norm: Optional[float] = None

    # Aggregation rule actually used ("sample_weighted" | "uniform")
    aggregation: Optional[str] = None

    # Communication split by direction and content (bytes, whole job)
    uplink_bytes: int = 0
    downlink_model_bytes: int = 0
    downlink_he_context_bytes: int = 0
    downlink_mask_bytes: int = 0

    # Top-K sparsification
    topk_mode: str = "none"
    topk_ratio: float = 1.0
    topk_coordinates: int = 0
    sparsification_time_sec: float = 0.0

    # Rounds in which each training client (partition index) participated
    client_participation: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_training_time_sec": round(self.total_training_time_sec, 3),
            "local_training_time_sec": round(self.local_training_time_sec, 3),
            "aggregation_time_sec": round(self.aggregation_time_sec, 3),
            "communication_time_sec": round(self.communication_time_sec, 3),
            "inference_time_sec": round(self.inference_time_sec, 3),
            "total_execution_time_sec": round(self.total_execution_time_sec, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "communication_kb_sent": round(self.communication_bytes_sent / 1024.0, 2),
            "communication_kb_received": round(self.communication_bytes_received / 1024.0, 2),
            "he_keygen_time_sec": round(self.he_keygen_time_sec, 4),
            "he_encryption_time_sec": round(self.he_encryption_time_sec, 4),
            "he_computation_time_sec": round(self.he_computation_time_sec, 4),
            "he_decryption_time_sec": round(self.he_decryption_time_sec, 4),
            "he_ciphertext_size_kb": round(self.he_ciphertext_size_bytes / 1024.0, 2),
            "dp_epsilon": self.dp_epsilon,
            "dp_delta": self.dp_delta,
            "dp_clipping_norm": self.dp_clipping_norm,
            "dp_noise_multiplier": self.dp_noise_multiplier,
            "dp_noise_location": self.dp_noise_location,
            "dp_epsilon_single_update_local": self.dp_epsilon_single_update_local,
            "dp_accountant": self.dp_accountant,
            "dp_clip_fraction": self.dp_clip_fraction,
            "dp_mean_preclip_norm": self.dp_mean_preclip_norm,
            "aggregation": self.aggregation,
            "uplink_bytes": self.uplink_bytes,
            "downlink_model_bytes": self.downlink_model_bytes,
            "downlink_he_context_bytes": self.downlink_he_context_bytes,
            "downlink_mask_bytes": self.downlink_mask_bytes,
            "topk_mode": self.topk_mode,
            "topk_ratio": self.topk_ratio,
            "topk_coordinates": self.topk_coordinates,
            "sparsification_time_sec": round(self.sparsification_time_sec, 4),
            "client_participation": {str(k): v for k, v in self.client_participation.items()},
        }


class MemoryProfiler:
    """Context manager for tracking peak memory in MB using tracemalloc."""

    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / (1024.0 * 1024.0)
        self.elapsed_sec = time.perf_counter() - self.start_time


def compute_privacy_overhead(
    target_time: float,
    baseline_time: float,
) -> float:
    """
    Overhead(%) = (T_privacy - T_baseline) / T_baseline * 100
    """
    if baseline_time <= 1e-9:
        return 0.0
    return float(((target_time - baseline_time) / baseline_time) * 100.0)
