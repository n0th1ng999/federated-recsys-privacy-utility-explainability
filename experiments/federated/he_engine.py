from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
    TENSEAL_VERSION = getattr(ts, "__version__", "unknown")
except ImportError:
    ts = None
    TENSEAL_AVAILABLE = False
    TENSEAL_VERSION = None


class HEEngine:
    """
    CKKS secure aggregation for Federated Learning (TenSEAL).

    Key roles (honest-but-curious model):
      * Key holder: generates the CKKS key pair and is the only party able to
        decrypt. It receives only the aggregated ciphertext. In the simulation it
        is a separate logical role from the aggregation server; the clients share
        the public context.
      * Aggregation server: holds `server_context` (public + evaluation keys, NO
        secret key) and can only add/scale ciphertexts.
    Confidentiality of individual updates against the server rests on the RLWE
    hardness assumption (IND-CPA security of CKKS) and on the server not colluding
    with the key holder. CKKS decryption of the aggregate is approximate.

    All ciphertext traffic is measured by serialising every chunk.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
        scale_bits: int = 40,
    ):
        if not TENSEAL_AVAILABLE:
            raise RuntimeError(
                "TenSEAL is not installed. HE experiments must run with real CKKS "
                "encryption; there is no simulated fallback."
            )
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = list(coeff_mod_bit_sizes)
        self.scale_bits = scale_bits
        self.scale = 2 ** scale_bits
        self.slot_count = self.poly_modulus_degree // 2

        t0 = time.perf_counter()
        self.secret_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        self.secret_context.global_scale = self.scale
        self.keygen_time_sec = time.perf_counter() - t0

        # What clients and the aggregation server receive: the public key only. No secret key,
        # and no Galois or relinearisation keys, which are needed only for rotations and
        # ciphertext-ciphertext products; FedAvg needs additions and plaintext scalings.
        public_bytes = self.secret_context.serialize(
            save_public_key=True, save_secret_key=False, save_galois_keys=False, save_relin_keys=False
        )
        self.public_context_bytes = len(public_bytes)
        self.server_context = ts.context_from(public_bytes)
        self.client_context = ts.context_from(public_bytes)

    # ------------------------------------------------------------------
    # Flattening helpers
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_weights(weights: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, ...], int]]]:
        """Flattens a model state dict / weight delta into a 1D float64 numpy array."""
        chunks = []
        shapes = []
        for key, tensor in weights.items():
            arr = tensor.detach().cpu().numpy().flatten().astype(np.float64)
            shapes.append((key, tuple(tensor.shape), len(arr)))
            chunks.append(arr)
        flat = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        return flat, shapes

    @staticmethod
    def unflatten_weights(
        flat: np.ndarray,
        shapes: List[Tuple[str, Tuple[int, ...], int]],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Reconstructs state dict / weight delta from a 1D flat numpy array."""
        weights = {}
        offset = 0
        for key, shape, length in shapes:
            chunk = flat[offset:offset + length]
            weights[key] = torch.from_numpy(chunk.reshape(shape)).float().to(device)
            offset += length
        return weights

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------
    def client_encrypt(self, flat_vector: np.ndarray) -> Tuple[List[bytes], float, int]:
        """
        Encrypts a vector with the public context, packing `slot_count` values per
        ciphertext. Returns (serialized_chunks, encryption_time_sec, total_bytes).
        """
        t0 = time.perf_counter()
        num_chunks = int(math.ceil(len(flat_vector) / self.slot_count))
        serialized = []
        for i in range(num_chunks):
            sub_vec = flat_vector[i * self.slot_count:(i + 1) * self.slot_count]
            serialized.append(ts.ckks_vector(self.client_context, sub_vec).serialize())
        enc_time = time.perf_counter() - t0
        return serialized, enc_time, int(sum(len(b) for b in serialized))

    # ------------------------------------------------------------------
    # Aggregation server (no secret key)
    # ------------------------------------------------------------------
    def server_aggregate(
        self,
        client_payloads: Sequence[List[bytes]],
        client_weights: Sequence[float],
    ) -> Tuple[List[bytes], float, int]:
        """
        Computes Enc(sum_k w_k * u_k) chunk by chunk using only the public context.
        Returns (serialized_aggregate, computation_time_sec, aggregate_bytes).
        """
        t0 = time.perf_counter()
        num_chunks = len(client_payloads[0])
        if any(len(p) != num_chunks for p in client_payloads):
            raise ValueError("Clients sent ciphertexts with different chunk layouts.")
        agg = []
        for i in range(num_chunks):
            acc = ts.ckks_vector_from(self.server_context, client_payloads[0][i]) * float(client_weights[0])
            for k in range(1, len(client_payloads)):
                acc += ts.ckks_vector_from(self.server_context, client_payloads[k][i]) * float(client_weights[k])
            agg.append(acc.serialize())
        comp_time = time.perf_counter() - t0
        return agg, comp_time, int(sum(len(b) for b in agg))

    # ------------------------------------------------------------------
    # Key holder
    # ------------------------------------------------------------------
    def key_holder_decrypt(self, aggregate: List[bytes], orig_len: int) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        dec = [np.array(ts.ckks_vector_from(self.secret_context, b).decrypt(), dtype=np.float64) for b in aggregate]
        out = np.concatenate(dec)[:orig_len]
        return out, time.perf_counter() - t0
