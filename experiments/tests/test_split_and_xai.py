"""Leakage checks for the temporal split and faithfulness of the XAI wrapper."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.data.movielens_adapter import ensure_movielens_data
from experiments.data.partitioner import custom_collate, prepare_experiment_data
from experiments.evaluation.xai_evaluator import TabularRecSysModelWrapper, extract_tabular_features_from_sample
from recommender_core.Core_Model import CoreHybridRecSys

DATA_DIR = os.path.join("data", "movielens")


@pytest.fixture(scope="module")
def bundle():
    ensure_movielens_data(DATA_DIR)
    return prepare_experiment_data(data_dir=DATA_DIR, num_clients=20, eval_users_count=50, seed=7)


def test_user_groups_are_disjoint(bundle):
    split = bundle.eval_data["split"]
    train, evals, nonmem = set(split["train_client_ids"]), set(split["eval_client_ids"]), set(split["mia_nonmember_user_ids"])
    assert not train & evals and not train & nonmem and not evals & nonmem


def test_eval_positives_are_future_and_unseen(bundle):
    ds = bundle.full_dataset
    cutoff = pd.Timestamp(bundle.eval_data["split"]["cutoff_date"])
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), parse_dates=["OrderDate"])
    items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv")).merge(orders, on="OrderID")
    for cid, idxs in bundle.eval_data["eval_user_samples"].items():
        user = items[items.ClientID == cid]
        before = set(user[user.OrderDate <= cutoff].ProductID)
        after = set(user[user.OrderDate > cutoff].ProductID)
        for i in idxs:
            pid = int(ds.sample_product_ids[i])
            if ds.sample_label_rank[i] > 0.5:
                assert pid in after and pid not in before
            else:
                assert pid not in before and pid not in after


def test_record_holdout_disjoint_from_training(bundle):
    for c, hold in bundle.eval_data["mia_record_holdout_samples"].items():
        assert not set(hold) & set(bundle.client_partitions[c])


def test_wrapper_equals_model_on_instance(bundle):
    torch.manual_seed(0)
    model = CoreHybridRecSys(branch_dims=bundle.branch_dims).eval()
    idx = next(iter(bundle.eval_data["eval_user_samples"].values()))[:5]
    for i in idx:
        sample = bundle.full_dataset[i]
        batch = custom_collate([sample])
        inputs = {g: {k: v.float() for k, v in batch[g].items()} for g in ("user_state", "collaborative", "pair_context")}
        with torch.no_grad():
            _, score, _ = model(inputs)
        wrapper = TabularRecSysModelWrapper(model, torch.device("cpu"), sample)
        x = extract_tabular_features_from_sample(sample)
        assert float(wrapper(x)[0]) == pytest.approx(float(score.flatten()[0]), abs=1e-5)
