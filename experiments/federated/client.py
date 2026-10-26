from __future__ import annotations

import copy
import time
import tracemalloc
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from recommender_core.Core_Model import CoreHybridRecSys
from recommender_core.train import pairwise_rank_loss, _derive_conf_target


class FederatedClient:
    """
    Simulates an individual federated client holding a local partition of interaction data.
    Trains locally and produces model weight deltas.
    """

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        branch_dims: Dict[str, Any],
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.branch_dims = branch_dims
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def train_epoch(
        self,
        global_weights: Dict[str, torch.Tensor],
        local_epochs: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """
        Loads global weights, executes local epochs, and returns:
        (weight_delta, num_samples, local_time_sec, peak_memory_mb).
        """
        tracemalloc.start()
        t0 = time.perf_counter()

        model = CoreHybridRecSys(branch_dims=self.branch_dims)
        model.load_state_dict(global_weights)
        model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_samples = 0

        for _ in range(local_epochs):
            for batch in self.dataloader:
                inputs = {
                    "user_state": {k: v.to(self.device) for k, v in batch["user_state"].items()},
                    "collaborative": {k: v.to(self.device) for k, v in batch["collaborative"].items()},
                    "pair_context": {k: v.to(self.device) for k, v in batch["pair_context"].items()},
                }
                labels_reg = batch["label_reg"].float().to(self.device)
                labels_rank = batch["label_rank"].float().to(self.device)
                user_idx = batch["user_idx"].long().to(self.device)

                batch_size = labels_reg.shape[0]
                total_samples += batch_size

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

        # Compute weight delta: Delta = W_local - W_global
        local_weights = model.state_dict()
        delta = {}
        for key in global_weights:
            delta[key] = (local_weights[key].cpu() - global_weights[key].cpu()).detach().clone()

        local_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024.0 * 1024.0)

        return delta, total_samples, local_time, peak_mb
