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
from recommender_core.dataset import (
    RecSysDatasetGeneric,
    SAMPLE_FEATURE_METRIC_NAMES,
    _concat_sample_column_blocks,
)


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


class HoldoutRecSysDataset(RecSysDatasetGeneric):
    """
    RecSysDatasetGeneric with a leakage-free evaluation block.

    Training users get the standard samples (interacted items + sampled negatives,
    all built from interactions up to the cutoff). Evaluation users get NO training
    samples; instead, for each one, the candidate set is
      * positives: items first purchased AFTER the global cutoff T (label 1), and
      * negatives: `candidates_per_user` items the user never interacted with,
        before or after T (label 0),
    with every feature computed from interactions up to T only.
    """

    def __init__(self, *args, eval_candidates: Dict[int, Tuple[List[int], List[int]]], split_seed: int, **kwargs):
        self._eval_candidates = eval_candidates
        self._split_seed = int(split_seed)
        super().__init__(*args, **kwargs)

    def _build_samples(self, num_workers, progress_callback=None):
        # Seed used by _build_samples_for_user for negative sampling of training users.
        self.seed = self._split_seed
        eval_set = set(self._eval_candidates)
        all_ids = self.client_ids
        self.client_ids = np.asarray([c for c in all_ids if int(c) not in eval_set])
        train_block = super()._build_samples(num_workers, progress_callback)
        self.client_ids = all_ids

        blocks = [train_block]
        for cid, (pos, neg) in self._eval_candidates.items():
            block = self._candidate_block(int(cid), pos, neg)
            if block is not None:
                blocks.append(block)
        return _concat_sample_column_blocks(blocks)

    def _candidate_block(self, cid: int, pos: List[int], neg: List[int]):
        user_signal = self.get_user_twin_state(cid)
        purchased = self.purchased_dict.get(cid, set())
        viewed = self.viewed_dict.get(cid, set()) - purchased
        state = self._get_user_pair_state(cid, user_signal=user_signal)
        cols = {"client_ids": [], "product_ids": [], "label_reg": [], "label_rank": [], "label_conf": []}
        metric_values = {name: [] for name in SAMPLE_FEATURE_METRIC_NAMES}
        for pid, label in [(p, 1.0) for p in pos] + [(n, 0.0) for n in neg]:
            if int(pid) not in self.prod_index_map:
                continue
            metrics = self._compute_pair_metrics(cid=cid, pid=int(pid), state=state, viewed=viewed)
            cols["client_ids"].append(cid)
            cols["product_ids"].append(int(pid))
            cols["label_reg"].append(label)
            cols["label_rank"].append(label)
            cols["label_conf"].append(1.0)
            for name in SAMPLE_FEATURE_METRIC_NAMES:
                metric_values[name].append(float(metrics.get(name, 0.0) or 0.0))
        if not cols["client_ids"]:
            return None
        out = {
            "client_ids": np.asarray(cols["client_ids"], dtype=np.int64),
            "product_ids": np.asarray(cols["product_ids"], dtype=np.int64),
            "label_reg": np.asarray(cols["label_reg"], dtype=np.float32),
            "label_rank": np.asarray(cols["label_rank"], dtype=np.float32),
            "label_conf": np.asarray(cols["label_conf"], dtype=np.float32),
        }
        for name in SAMPLE_FEATURE_METRIC_NAMES:
            out[f"metric_{name}"] = np.asarray(metric_values[name], dtype=np.float32)
        return out


def _explode_view_products(views_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in views_df.itertuples(index=False):
        for pid in str(getattr(row, "ProductIDs", "")).split(","):
            pid = pid.strip()
            if pid.isdigit():
                rows.append((int(row.ClientID), int(pid)))
    return pd.DataFrame(rows, columns=["ClientID", "ProductID"])


def prepare_experiment_data(
    data_dir: str = "data",
    num_clients: int = 20,
    eval_users_count: int = 50,
    candidates_per_user: int = 50,
    seed: int = 42,
    sample_build_workers: int = 1,
    cutoff_quantile: float = 0.8,
    mia_record_holdout: float = 0.2,
    mia_nonmember_users: int = 20,
    min_history_items: int = 3,
) -> ExperimentDataBundle:
    """
    Leakage-free temporal split.

    1. Global cutoff T = `cutoff_quantile` of all purchase dates. Every feature,
       catalog statistic and training sample uses only interactions dated <= T;
       "now" for recency features is pinned to T.
    2. Users eligible for evaluation have >= `min_history_items` purchased items
       before T and >= 1 new item purchased after T. `eval_users_count` of them are
       drawn at random (seeded); `num_clients` training users are drawn from the
       remaining users with history before T. Training and evaluation users are
       disjoint.
    3. For each training client, a random `mia_record_holdout` fraction of its
       samples is withheld from training and used only as same-user non-members
       in the record-level membership-inference attack.
    4. `mia_nonmember_users` further users, drawn from the same pool as the
       training clients and processed identically (same sample construction), never
       train. They are the non-members of the client-participation attack.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_clients = pd.read_csv(os.path.join(data_dir, "clientes.csv"))
    orders_all = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    order_items_all = pd.read_csv(os.path.join(data_dir, "order_items.csv"))
    products_df = pd.read_csv(os.path.join(data_dir, "produtos.csv"))
    views_all = pd.read_csv(os.path.join(data_dir, "viewed_products.csv"))
    price_csv = os.path.join(data_dir, "price_history.csv")
    relations_csv = os.path.join(data_dir, "product_relations.csv")
    price_history_df = pd.read_csv(price_csv) if os.path.exists(price_csv) else None
    product_relations_df = pd.read_csv(relations_csv) if os.path.exists(relations_csv) else None

    orders_all["_date"] = pd.to_datetime(orders_all["OrderDate"], errors="coerce")
    views_all["_date"] = pd.to_datetime(views_all["ViewDate"], errors="coerce")
    purchases = order_items_all[["OrderID", "ProductID"]].merge(
        orders_all[["OrderID", "ClientID", "_date"]], on="OrderID"
    )
    cutoff = purchases["_date"].quantile(cutoff_quantile)

    pre = purchases[purchases["_date"] <= cutoff]
    post = purchases[purchases["_date"] > cutoff]
    pre_items = pre.groupby("ClientID")["ProductID"].apply(set).to_dict()
    post_items = post.groupby("ClientID")["ProductID"].apply(set).to_dict()
    pre_views = _explode_view_products(views_all[views_all["_date"] <= cutoff])
    pre_view_items = pre_views.groupby("ClientID")["ProductID"].apply(set).to_dict()
    all_view_items = _explode_view_products(views_all).groupby("ClientID")["ProductID"].apply(set).to_dict()

    known_clients = set(all_clients["ClientID"].tolist())
    eval_eligible = sorted(
        cid for cid, items in post_items.items()
        if cid in known_clients
        and len(pre_items.get(cid, set())) >= min_history_items
        and len(items - pre_items.get(cid, set()) - pre_view_items.get(cid, set())) >= 1
    )
    if len(eval_eligible) < eval_users_count:
        raise ValueError(f"Only {len(eval_eligible)} users are eligible for temporal evaluation; need {eval_users_count}.")
    eval_client_ids = sorted(int(c) for c in rng.choice(eval_eligible, size=eval_users_count, replace=False))

    train_eligible = sorted(
        cid for cid, items in pre_items.items()
        if cid in known_clients and cid not in set(eval_client_ids) and len(items) >= min_history_items
    )
    if len(train_eligible) < num_clients:
        raise ValueError(f"Only {len(train_eligible)} users are eligible for training; need {num_clients}.")
    if len(train_eligible) < num_clients + mia_nonmember_users:
        raise ValueError(f"Only {len(train_eligible)} users are eligible for training; need {num_clients + mia_nonmember_users}.")
    drawn = [int(c) for c in rng.choice(train_eligible, size=num_clients + mia_nonmember_users, replace=False)]
    train_client_ids = sorted(drawn[:num_clients])
    nonmember_user_ids = sorted(drawn[num_clients:])
    assert not set(train_client_ids) & set(eval_client_ids), "train/eval users overlap"
    assert not set(nonmember_user_ids) & (set(train_client_ids) | set(eval_client_ids)), "MIA non-members overlap"

    client_ids = train_client_ids + nonmember_user_ids + eval_client_ids
    client_id_set = set(client_ids)
    selected_clients = all_clients[all_clients["ClientID"].isin(client_id_set)].copy()

    # Only interactions up to the cutoff enter preprocessing (features + catalog statistics).
    orders_df = orders_all[orders_all["ClientID"].isin(client_id_set) & (orders_all["_date"] <= cutoff)].drop(columns="_date")
    order_items_df = order_items_all[order_items_all["OrderID"].isin(set(orders_df["OrderID"]))].copy()
    views_df = views_all[views_all["ClientID"].isin(client_id_set) & (views_all["_date"] <= cutoff)].drop(columns="_date")
    if price_history_df is not None and "Date" in price_history_df.columns and not price_history_df.empty:
        price_history_df = price_history_df[pd.to_datetime(price_history_df["Date"], errors="coerce") <= cutoff].copy()

    os.environ["RECSYS_REFERENCE_TIME"] = pd.Timestamp(cutoff).isoformat()

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

    # Evaluation candidates: future new purchases vs. never-interacted items.
    catalog = np.asarray(sorted(int(p) for p in products_df["ProductID"].tolist()), dtype=np.int64)
    eval_candidates: Dict[int, Tuple[List[int], List[int]]] = {}
    for cid in eval_client_ids:
        history = pre_items.get(cid, set()) | pre_view_items.get(cid, set())
        positives = sorted(int(p) for p in (post_items.get(cid, set()) - history))
        ever = history | post_items.get(cid, set()) | all_view_items.get(cid, set())
        pool = catalog[~np.isin(catalog, np.asarray(sorted(ever), dtype=np.int64))]
        negatives = sorted(int(p) for p in rng.choice(pool, size=min(candidates_per_user, pool.size), replace=False))
        eval_candidates[cid] = (positives, negatives)

    full_dataset = HoldoutRecSysDataset(
        clients_df=proc_clients_df,
        orders_df=proc_orders_df,
        order_items_df=proc_order_items_df,
        products_df=proc_products_df,
        views_df=proc_views_df,
        signals=signals,
        max_seq_len=25,
        num_workers=1,
        sample_build_workers=sample_build_workers,
        eval_candidates=eval_candidates,
        split_seed=seed,
    )

    # Partition training samples by client and withhold a per-client record holdout for MIA.
    train_cid_to_client_idx = {cid: idx for idx, cid in enumerate(train_client_ids)}
    all_client_samples: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    eval_user_samples: Dict[int, List[int]] = {cid: [] for cid in eval_client_ids}
    nonmember_user_samples: Dict[int, List[int]] = {cid: [] for cid in nonmember_user_ids}
    for s_idx in range(len(full_dataset)):
        cid = int(full_dataset.sample_client_ids[s_idx])
        if cid in train_cid_to_client_idx:
            all_client_samples[train_cid_to_client_idx[cid]].append(s_idx)
        elif cid in eval_user_samples:
            eval_user_samples[cid].append(s_idx)
        elif cid in nonmember_user_samples:
            nonmember_user_samples[cid].append(s_idx)

    client_partitions: Dict[int, List[int]] = {}
    mia_record_holdout_samples: Dict[int, List[int]] = {}
    for idx, samples in all_client_samples.items():
        perm = [samples[i] for i in rng.permutation(len(samples))]
        n_hold = int(round(mia_record_holdout * len(perm)))
        mia_record_holdout_samples[idx] = sorted(perm[:n_hold])
        client_partitions[idx] = sorted(perm[n_hold:])
        if not client_partitions[idx]:
            raise ValueError(f"Training client {idx} has no samples after the record holdout.")

    valid_eval_users = [
        cid for cid, idxs in eval_user_samples.items()
        if any(full_dataset.sample_label_rank[i] > 0.5 for i in idxs)
    ]
    eval_data = {
        "eval_user_ids": valid_eval_users,
        "eval_user_samples": {cid: eval_user_samples[cid] for cid in valid_eval_users},
        "relevance_threshold": 0.5,
        "split": {
            "type": "global_temporal_holdout",
            "cutoff_date": str(pd.Timestamp(cutoff).date()),
            "cutoff_quantile": cutoff_quantile,
            "train_client_ids": train_client_ids,
            "eval_client_ids": eval_client_ids,
            "num_eval_positives": int(sum(len(p) for p, _ in eval_candidates.values())),
            "negatives_per_user": candidates_per_user,
            "mia_record_holdout": mia_record_holdout,
            "mia_nonmember_user_ids": nonmember_user_ids,
        },
        "mia_record_holdout_samples": mia_record_holdout_samples,
        "mia_nonmember_user_samples": {c: v for c, v in nonmember_user_samples.items() if v},
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
