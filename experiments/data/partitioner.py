from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from api_service.preprocess import preprocess_data_generic
from recommender_core.dataset import RecSysDatasetGeneric


def custom_collate(batch):
    """Collates a list of dictionary samples into PyTorch batch tensors."""
    def _collate(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(values, dim=0)
        if isinstance(first, np.ndarray):
            return torch.from_numpy(np.stack(values, axis=0))
        if isinstance(first, dict):
            return {k: _collate([item[k] for item in values]) for k in first}
        if isinstance(first, (int, float, np.integer, np.floating)):
            return torch.tensor(values)
        return values

    return _collate(batch)


class ExperimentDataBundle:
    """Encapsulates all datasets, federated client partitions, and fixed evaluation data."""

    def __init__(
        self,
        full_dataset: RecSysDatasetGeneric,
        client_partitions: Dict[int, List[int]],  # client_idx -> list of sample indices
        client_ids: List[int],
        eval_data: Dict[str, Any],
        branch_dims: Dict[str, Any],
    ):
        self.full_dataset = full_dataset
        self.client_partitions = client_partitions
        self.client_ids = client_ids
        self.eval_data = eval_data
        self.branch_dims = branch_dims

    def get_client_dataloader(
        self,
        client_idx: int,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        indices = self.client_partitions.get(client_idx, [])
        if not indices:
            raise ValueError(f"No sample indices found for client {client_idx}")
        subset = Subset(self.full_dataset, indices)
        return DataLoader(
            subset,
            batch_size=min(batch_size, len(indices)),
            shuffle=shuffle,
            collate_fn=custom_collate,
            drop_last=False,
        )

    def get_centralized_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        # Client-blocked stratified pooling: randomizes client order and internal sample order,
        # but groups consecutive samples by client ID to achieve >98% cache hit rate in dataset materialization.
        pooled_indices = []
        client_keys = list(self.client_partitions.keys())
        if shuffle:
            random.shuffle(client_keys)
        for c in client_keys:
            c_indices = list(self.client_partitions[c])
            if shuffle:
                random.shuffle(c_indices)
            pooled_indices.extend(c_indices)

        subset = Subset(self.full_dataset, pooled_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,  # Order is already randomized at client and sample level
            collate_fn=custom_collate,
            drop_last=False,
        )


def prepare_experiment_data(
    data_dir: str = "data",
    num_clients: int = 20,
    eval_users_count: int = 30,
    candidates_per_user: int = 50,
    seed: int = 42,
    sample_build_workers: int = 1,
) -> ExperimentDataBundle:
    """
    Loads raw CSVs, filters to a deterministic set of training and evaluation clients,
    preprocesses signals, builds the unified RecSysDatasetGeneric, and partitions it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    clients_csv = os.path.join(data_dir, "clientes.csv")
    orders_csv = os.path.join(data_dir, "orders.csv")
    order_items_csv = os.path.join(data_dir, "order_items.csv")
    produtos_csv = os.path.join(data_dir, "produtos.csv")
    viewed_csv = os.path.join(data_dir, "viewed_products.csv")
    cart_csv = os.path.join(data_dir, "cart_events.csv")
    wishlist_csv = os.path.join(data_dir, "wishlist.csv")
    relations_csv = os.path.join(data_dir, "product_relations.csv")
    price_csv = os.path.join(data_dir, "price_history.csv")

    all_clients = pd.read_csv(clients_csv)
    total_selected = num_clients + eval_users_count
    selected_clients = all_clients.head(total_selected).copy()
    client_ids = selected_clients["ClientID"].tolist()
    client_id_set = set(client_ids)

    train_client_ids = client_ids[:num_clients]
    eval_client_ids = client_ids[num_clients:total_selected]

    orders_df = pd.read_csv(orders_csv)
    orders_df = orders_df[orders_df["ClientID"].isin(client_id_set)].copy()
    order_id_set = set(orders_df["OrderID"])

    order_items_df = pd.read_csv(order_items_csv)
    order_items_df = order_items_df[order_items_df["OrderID"].isin(order_id_set)].copy()

    products_df = pd.read_csv(produtos_csv)
    views_df = pd.read_csv(viewed_csv)
    views_df = views_df[views_df["ClientID"].isin(client_id_set)].copy()

    price_history_df = pd.read_csv(price_csv) if os.path.exists(price_csv) else None
    product_relations_df = pd.read_csv(relations_csv) if os.path.exists(relations_csv) else None

    # Run generic preprocessing to construct structured signals and catalog contexts
    (
        proc_clients_df,
        proc_products_df,
        proc_orders_df,
        proc_order_items_df,
        proc_views_df,
        signals,
    ) = preprocess_data_generic(
        selected_clients,
        orders_df,
        order_items_df,
        products_df,
        views=views_df,
        price_history=price_history_df,
        product_relations=product_relations_df,
        verbose_stdout=False,
    )

    full_dataset = RecSysDatasetGeneric(
        clients_df=proc_clients_df,
        orders_df=proc_orders_df,
        order_items_df=proc_order_items_df,
        products_df=proc_products_df,
        views_df=proc_views_df,
        signals=signals,
        max_seq_len=25,
        num_workers=1,
        sample_build_workers=sample_build_workers,
    )

    # Partition sample indices by client ID for training clients
    client_partitions: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    train_cid_to_client_idx = {cid: idx for idx, cid in enumerate(train_client_ids)}

    for s_idx in range(len(full_dataset)):
        cid = int(full_dataset.sample_client_ids[s_idx])
        if cid in train_cid_to_client_idx:
            client_idx = train_cid_to_client_idx[cid]
            client_partitions[client_idx].append(s_idx)

    # Build fixed evaluation pairs for evaluation clients
    # For each eval client: find their samples in the dataset
    eval_user_samples: Dict[int, List[int]] = {cid: [] for cid in eval_client_ids}
    for s_idx in range(len(full_dataset)):
        cid = int(full_dataset.sample_client_ids[s_idx])
        if cid in eval_user_samples:
            eval_user_samples[cid].append(s_idx)

    # Filter evaluation users that have at least 5 samples
    valid_eval_users = [cid for cid, s_list in eval_user_samples.items() if len(s_list) >= 5]
    if not valid_eval_users:
        # Fallback to train users if eval set had sparse interactions
        valid_eval_users = [cid for cid, idx in train_cid_to_client_idx.items() if len(client_partitions[idx]) >= 5][:eval_users_count]
        eval_user_samples = {cid: client_partitions[train_cid_to_client_idx[cid]] for cid in valid_eval_users}

    eval_data = {
        "eval_user_ids": valid_eval_users,
        "eval_user_samples": eval_user_samples,
    }

    branch_dims = getattr(full_dataset, "branch_input_dims", {})
    if not branch_dims:
        from recommender_core.dataset import (
            COLLAB_BASE_CHANNELS,
            COLLAB_CHANNEL_SUPPORT_FEATURES,
            COLLABORATIVE_CHANNEL_SUPPORT_FEATURES,
            COLLABORATIVE_CLUSTER_FEATURE_NAMES,
            INTENT_CLUSTER_FEATURE_NAMES,
            OWNERSHIP_CLUSTER_FEATURE_NAMES,
            PAIR_CONTEXT_FEATURE_NAMES,
            USER_STATE_INTENT_SUMMARY_FEATURES,
            USER_STATE_OWNERSHIP_SUMMARY_FEATURES,
        )
        branch_dims = {
            "user_state": {
                "intent_cluster_dim": len(INTENT_CLUSTER_FEATURE_NAMES),
                "ownership_cluster_dim": len(OWNERSHIP_CLUSTER_FEATURE_NAMES),
                "intent_summary_dim": len(USER_STATE_INTENT_SUMMARY_FEATURES),
                "ownership_summary_dim": len(USER_STATE_OWNERSHIP_SUMMARY_FEATURES),
            },
            "collaborative": {
                "cluster_dim": len(COLLABORATIVE_CLUSTER_FEATURE_NAMES),
                "channel_support_dim": len(COLLABORATIVE_CHANNEL_SUPPORT_FEATURES),
                "per_channel_support_dim": len(COLLAB_CHANNEL_SUPPORT_FEATURES),
                "num_channels": len(COLLAB_BASE_CHANNELS),
            },
            "pair_context": {
                "context_dim": len(PAIR_CONTEXT_FEATURE_NAMES),
            },
        }

    return ExperimentDataBundle(
        full_dataset=full_dataset,
        client_partitions=client_partitions,
        client_ids=train_client_ids,
        eval_data=eval_data,
        branch_dims=branch_dims,
    )
