import concurrent.futures
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from support.feature_contract import (
    COLLAB_BASE_CHANNELS,
    COLLAB_CHANNEL_SUPPORT_FEATURES,
    COLLABORATIVE_CHANNEL_SUPPORT_FEATURES,
    COLLABORATIVE_CLUSTER_TOP_K,
    COLLABORATIVE_CLUSTER_FEATURE_NAMES,
    INTENT_CLUSTER_FEATURE_NAMES,
    OWNERSHIP_CLUSTER_FEATURE_NAMES,
    PAIR_CONTEXT_FEATURE_NAMES,
    PURCHASE_CONFIDENCE_WEIGHTS,
    PURCHASE_FAMILY_INTENT_WEIGHTS,
    PURCHASE_LOGIT_SCALE,
    PURCHASE_LOGIT_WEIGHTS,
    PURCHASE_PRODUCT_FIT_WEIGHTS,
    PURCHASE_REPLENISHMENT_INTENT_WEIGHTS,
    SCORE_ATTR_KEYS,
    SIGNAL_POLICIES,
    USER_STATE_INTENT_SUMMARY_FEATURES,
    USER_STATE_OWNERSHIP_SUMMARY_FEATURES,
)
from support.signal_utils import (
    cacheable_attr_value as _cacheable_attr_value,
    clip as _clip,
    derive_collaborative_channel_support as _derive_collaborative_channel_support,
    exp_decay as _exp_decay,
    is_real_source_token as _is_real_source_token,
    normalize_attr_token as _normalize_attr_token,
    normalize_attr_tokens as _normalize_attr_tokens,
    normalize_lookup as _normalize_lookup,
    safe_float as _safe_float,
    sigmoid as _sigmoid,
    split_product_ids as _split_product_ids,
)
from support.twin_state import (
    clone_user_twin_state,
    extract_collaborative_context,
    extract_user_domain,
    is_user_twin_state,
    make_empty_user_twin_state,
    to_utc_timestamp,
)

DEFAULT_NEG_RATIO = 0.6
MIN_NEGATIVE_SAMPLES = 20
RATING_FALLBACK_VALUE = 3.5
SAMPLE_FEATURE_METRIC_NAMES = tuple(PAIR_CONTEXT_FEATURE_NAMES)


_DATASET_BUILD_WORKER = None


def _fill_frame_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if numeric_cols:
        out.loc[:, numeric_cols] = out[numeric_cols].fillna(0)
    other_cols = [col for col in out.columns if col not in numeric_cols]
    if other_cols:
        out.loc[:, other_cols] = out[other_cols].where(out[other_cols].notna(), "")
    return out


def _empty_sample_columns():
    out = {
        "client_ids": np.empty((0,), dtype=np.int64),
        "product_ids": np.empty((0,), dtype=np.int64),
        "label_reg": np.empty((0,), dtype=np.float32),
        "label_rank": np.empty((0,), dtype=np.float32),
        "label_conf": np.empty((0,), dtype=np.float32),
    }
    for name in SAMPLE_FEATURE_METRIC_NAMES:
        out[f"metric_{name}"] = np.empty((0,), dtype=np.float32)
    return out


def _concat_sample_column_blocks(blocks):
    valid_blocks = [
        block
        for block in blocks
        if isinstance(block, dict)
        and int(block.get("client_ids", np.empty((0,), dtype=np.int64)).shape[0]) > 0
    ]
    if not valid_blocks:
        return _empty_sample_columns()
    keys = (
        "client_ids",
        "product_ids",
        "label_reg",
        "label_rank",
        "label_conf",
        *(f"metric_{name}" for name in SAMPLE_FEATURE_METRIC_NAMES),
    )
    return {
        key: np.concatenate([np.asarray(block[key]) for block in valid_blocks], axis=0)
        for key in keys
    }


def _chunk_sequence(values, chunk_size):
    size = max(1, int(chunk_size))
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _purchase_target_from_state_metrics(state, metrics, rank_preference):
    branches = state.get("branches", {}) if isinstance(state, dict) else {}
    user_state = branches.get("user_state", {}) if isinstance(branches, dict) else {}
    intent_summary = user_state.get("intent", {}).get("summary", {}) if isinstance(user_state.get("intent", {}), dict) else {}
    ownership_summary = user_state.get("ownership", {}).get("summary", {}) if isinstance(user_state.get("ownership", {}), dict) else {}

    interest_strength = _clip(intent_summary.get("interest_strength", 0.0))
    source_view_share = _clip(intent_summary.get("source_view_share", 0.0))
    active_view_mass = _clip(intent_summary.get("active_view_mass", 0.0))
    active_wishlist_mass = _clip(intent_summary.get("active_wishlist_mass", 0.0))
    active_cart_mass = _clip(intent_summary.get("active_cart_mass", 0.0))
    focus_concentration = _clip(intent_summary.get("focus_concentration", 0.0))

    ownership_interest_ratio = _clip(ownership_summary.get("ownership_interest_ratio", 0.0))
    budget_coverage = _clip(ownership_summary.get("budget_coverage", 0.0))

    intent_score = _clip(metrics.get("intent_score", 0.0))
    source_intent_score = _clip(metrics.get("source_intent_score", 0.0))
    source_family_recovery = _clip(metrics.get("source_family_recovery", 0.0))
    source_purchase_recency = _clip(metrics.get("source_purchase_recency", 0.0))
    expected_cycle_norm = _clip(metrics.get("expected_cycle_norm", 0.0))
    user_cycle_ratio = _clip(metrics.get("user_cycle_ratio", 0.0))
    user_faster_than_expected = _clip(metrics.get("user_faster_than_expected", 0.0))
    user_slower_than_expected = _clip(metrics.get("user_slower_than_expected", 0.0))
    same_item_penalty = _clip(metrics.get("same_item_penalty", 0.0))
    substitute_penalty = _clip(metrics.get("substitute_penalty", 0.0))
    cluster_suppression = _clip(metrics.get("cluster_suppression", 0.0))
    brand_match = _clip(metrics.get("brand_match", 0.0))
    keyword_match = _clip(metrics.get("keyword_match", 0.0))
    size_match = _clip(metrics.get("size_match", 0.0))
    color_match = _clip(metrics.get("color_match", 0.0))
    material_match = _clip(metrics.get("material_match", 0.0))
    source_purchase_score = _clip(metrics.get("source_purchase_score", 0.0))
    source_budget_alignment = _clip(metrics.get("source_budget_alignment", 0.0))
    global_budget_alignment = _clip(metrics.get("global_budget_alignment", 0.0))
    price_norm = _clip(metrics.get("price_norm", 0.0))
    intent_momentum = _clip(metrics.get("intent_momentum", 0.0))
    interaction_boost = _clip(metrics.get("interaction_boost", 0.0))

    behavioral_intent = _clip(
        0.28 * interest_strength
        + 0.16 * source_view_share
        + 0.14 * active_view_mass
        + 0.18 * active_wishlist_mass
        + 0.26 * active_cart_mass
        + 0.08 * focus_concentration
    )
    replenishment_intent = _clip(
        PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["source_family_recovery"] * source_family_recovery
        + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["source_purchase_recency"] * source_purchase_recency
        + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["expected_cycle_norm"] * expected_cycle_norm
        + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["user_faster_than_expected"] * user_faster_than_expected
        + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["user_slower_than_expected"] * user_slower_than_expected
        + PURCHASE_REPLENISHMENT_INTENT_WEIGHTS["user_cycle_ratio"] * user_cycle_ratio
    )
    family_intent = _clip(
        PURCHASE_FAMILY_INTENT_WEIGHTS["behavioral_intent"] * behavioral_intent
        + PURCHASE_FAMILY_INTENT_WEIGHTS["intent_score"] * intent_score
        + PURCHASE_FAMILY_INTENT_WEIGHTS["source_intent_score"] * source_intent_score
        + PURCHASE_FAMILY_INTENT_WEIGHTS["intent_momentum"] * intent_momentum
        + PURCHASE_FAMILY_INTENT_WEIGHTS["interaction_boost"] * interaction_boost
        + PURCHASE_FAMILY_INTENT_WEIGHTS["replenishment_intent"] * replenishment_intent
        + PURCHASE_FAMILY_INTENT_WEIGHTS["same_item_relief"] * (1.0 - same_item_penalty)
        + PURCHASE_FAMILY_INTENT_WEIGHTS["substitute_relief"] * (1.0 - substitute_penalty)
        + PURCHASE_FAMILY_INTENT_WEIGHTS["cluster_relief"] * (1.0 - cluster_suppression)
        + 0.04 * ownership_interest_ratio
        + 0.03 * budget_coverage
    )
    product_fit = _clip(
        PURCHASE_PRODUCT_FIT_WEIGHTS["brand_match"] * brand_match
        + PURCHASE_PRODUCT_FIT_WEIGHTS["keyword_match"] * keyword_match
        + PURCHASE_PRODUCT_FIT_WEIGHTS["size_match"] * size_match
        + PURCHASE_PRODUCT_FIT_WEIGHTS["color_match"] * color_match
        + PURCHASE_PRODUCT_FIT_WEIGHTS["material_match"] * material_match
        + PURCHASE_PRODUCT_FIT_WEIGHTS["source_purchase_score"] * source_purchase_score
        + PURCHASE_PRODUCT_FIT_WEIGHTS["source_budget_alignment"] * source_budget_alignment
        + PURCHASE_PRODUCT_FIT_WEIGHTS["global_budget_alignment"] * global_budget_alignment
        + PURCHASE_PRODUCT_FIT_WEIGHTS["price_norm"] * price_norm
    )

    logit = PURCHASE_LOGIT_WEIGHTS["bias"]
    logit += PURCHASE_LOGIT_WEIGHTS["family_intent"] * family_intent
    logit += PURCHASE_LOGIT_WEIGHTS["product_fit"] * product_fit
    intent_fit_synergy = family_intent * product_fit
    intent_fit_compound = intent_fit_synergy * (0.5 * (family_intent + product_fit))
    logit += PURCHASE_LOGIT_WEIGHTS["intent_fit_synergy"] * intent_fit_synergy
    logit += PURCHASE_LOGIT_WEIGHTS["intent_fit_compound"] * intent_fit_compound
    logit *= PURCHASE_LOGIT_SCALE

    signal_conf = (
        PURCHASE_CONFIDENCE_WEIGHTS["family_intent"] * family_intent
        + PURCHASE_CONFIDENCE_WEIGHTS["product_fit"] * product_fit
        + PURCHASE_CONFIDENCE_WEIGHTS["intent_fit_synergy"] * intent_fit_synergy
        + PURCHASE_CONFIDENCE_WEIGHTS["intent_fit_compound"] * intent_fit_compound
        + PURCHASE_CONFIDENCE_WEIGHTS["rank_preference"] * rank_preference
    )
    calibrated_conf = _clip(signal_conf ** 0.72)
    label_conf = _clip(0.12 + 0.88 * calibrated_conf)
    label_reg = _clip(_sigmoid(logit), 0.01, 0.99)
    return float(label_reg), float(label_conf)


def _init_dataset_build_worker(dataset_obj):
    global _DATASET_BUILD_WORKER
    _DATASET_BUILD_WORKER = dataset_obj


def _build_samples_for_user(dataset_obj, cid):
    client_ids = []
    product_ids = []
    label_reg_values = []
    label_rank_values = []
    label_conf_values = []
    metric_values = {name: [] for name in SAMPLE_FEATURE_METRIC_NAMES}
    user_signal = dataset_obj.get_user_twin_state(cid)
    if not user_signal:
        return None

    purchased = dataset_obj.purchased_dict.get(cid, set())
    viewed = dataset_obj.viewed_dict.get(cid, set()) - purchased
    interacted = purchased | viewed
    if not interacted:
        return None

    interacted_ids = [
        int(pid)
        for pid in interacted
        if int(pid) in dataset_obj.prod_index_map
    ]
    if not interacted_ids:
        return None

    all_prod_ids = dataset_obj.prod_ids_array
    interacted_arr = np.asarray(interacted_ids, dtype=np.int64)
    interacted_indices = np.fromiter(
        (int(dataset_obj.prod_index_map[pid]) for pid in interacted_ids),
        dtype=np.int64,
        count=len(interacted_ids),
    )
    keep_mask = np.ones(all_prod_ids.shape[0], dtype=bool)
    keep_mask[interacted_indices] = False
    non_interacted = all_prod_ids[keep_mask]
    if non_interacted.size == 0:
        return None

    user_seed = (dataset_obj.seed * 1000003 + int(cid) * 9176) % (2 ** 32 - 1)
    rng = np.random.default_rng(user_seed)

    neg_count = int(
        min(
            non_interacted.size,
            max(int(len(interacted) * DEFAULT_NEG_RATIO), MIN_NEGATIVE_SAMPLES),
        )
    )
    sampled_neg = (
        rng.choice(non_interacted, size=neg_count, replace=False)
        if neg_count > 0
        else np.empty(0, dtype=np.int64)
    )

    pos_items = interacted_arr
    label_items = np.concatenate([pos_items, sampled_neg])

    state = dataset_obj._get_user_pair_state(cid, user_signal=user_signal)
    for pid in label_items:
        pid = int(pid)
        metrics = dataset_obj._compute_pair_metrics(
            cid=cid,
            pid=pid,
            state=state,
            viewed=viewed,
        )
        label_rank = dataset_obj._label_rank(
            cid=cid,
            pid=pid,
            state=state,
            viewed=viewed,
            pair_metrics=metrics,
        )

        rank_preference = _clip((_safe_float(label_rank, RATING_FALLBACK_VALUE) - 1.0) / 4.0)
        label_reg, label_conf = _purchase_target_from_state_metrics(
            state=state,
            metrics=metrics,
            rank_preference=rank_preference,
        )

        client_ids.append(int(cid))
        product_ids.append(int(pid))
        label_reg_values.append(float(label_reg))
        label_rank_values.append(float(label_rank))
        label_conf_values.append(float(label_conf))
        for name in SAMPLE_FEATURE_METRIC_NAMES:
            metric_values[name].append(float(_safe_float(metrics.get(name, 0.0), 0.0)))

    if not client_ids:
        return None
    out = {
        "client_ids": np.asarray(client_ids, dtype=np.int64),
        "product_ids": np.asarray(product_ids, dtype=np.int64),
        "label_reg": np.asarray(label_reg_values, dtype=np.float32),
        "label_rank": np.asarray(label_rank_values, dtype=np.float32),
        "label_conf": np.asarray(label_conf_values, dtype=np.float32),
    }
    for name in SAMPLE_FEATURE_METRIC_NAMES:
        out[f"metric_{name}"] = np.asarray(metric_values[name], dtype=np.float32)
    return out


def _build_samples_chunk(client_ids):
    dataset_obj = _DATASET_BUILD_WORKER
    blocks = []
    for cid in client_ids:
        res = _build_samples_for_user(dataset_obj, cid)
        if res is not None:
            blocks.append(res)
    return _concat_sample_column_blocks(blocks)


class RecSysDatasetGeneric(Dataset):

    def __init__(
        self,
        clients_df,
        orders_df,
        order_items_df,
        products_df,
        views_df=None,
        signals=None,
        max_seq_len=25,
        num_workers: int = 4,
        sample_build_workers: int | None = None,
        progress_callback=None,
    ):
        self.max_seq_len = max_seq_len
        self.collab_top_k = min(int(max_seq_len), int(COLLABORATIVE_CLUSTER_TOP_K))
        signals_map = signals or {}
        self.seed = 42
        self.num_workers = max(1, int(num_workers))
        self.sample_build_workers = max(
            1,
            int(sample_build_workers if sample_build_workers is not None else self.num_workers),
        )

        self.clients_df = _fill_frame_missing_values(clients_df)
        self.products_df = _fill_frame_missing_values(products_df)
        self.orders_df = orders_df.copy()
        self.order_items_df = order_items_df.copy()
        self.views_df = views_df.copy() if views_df is not None else pd.DataFrame()

        self.client_ids = self.clients_df["ClientID"].to_numpy()
        self.prod_ids = self.products_df["ProductID"].to_numpy()
        self.client_index_map = {cid: i for i, cid in enumerate(self.client_ids)}
        self.prod_index_map = {pid: i for i, pid in enumerate(self.prod_ids)}
        self.catalog_context = signals_map.get("_catalog_context", {}) if isinstance(signals_map, dict) else {}
        family_context = self.catalog_context.get("family", {}) if isinstance(self.catalog_context, dict) else {}
        raw_family_lookup = family_context.get("product_family_lookup", {}) if isinstance(family_context, dict) else {}
        self.product_family_lookup = {
            int(pid): str(fid)
            for pid, fid in (raw_family_lookup.items() if isinstance(raw_family_lookup, dict) else [])
            if str(pid).isdigit() and fid is not None
        }
        raw_family_meta = family_context.get("metadata", {}) if isinstance(family_context, dict) else {}
        self.family_meta_lookup = {
            str(fid): dict(meta)
            for fid, meta in (raw_family_meta.items() if isinstance(raw_family_meta, dict) else [])
            if isinstance(meta, dict)
        }
        raw_taxonomy_lookup = family_context.get("taxonomy_leaf_lookup", {}) if isinstance(family_context, dict) else {}
        self.product_taxonomy_leaf_lookup = {
            int(pid): str(leaf_key)
            for pid, leaf_key in (raw_taxonomy_lookup.items() if isinstance(raw_taxonomy_lookup, dict) else [])
            if str(pid).isdigit() and leaf_key is not None
        }

        products_indexed = self.products_df.set_index("ProductID", drop=False)
        self.cat_lookup = self._build_product_attr_lookups()
        self.product_attrs = {
            int(pid): (
                self.cat_lookup["Brand"].get(pid),
                self.cat_lookup["Category"].get(pid),
                self.cat_lookup["Subcategory"].get(pid),
                self.cat_lookup["Size"].get(pid),
                self.cat_lookup["Color"].get(pid),
                self.cat_lookup["Material"].get(pid),
                self.cat_lookup["Keyword"].get(pid),
            )
            for pid in self.prod_ids
        }
        self.empty_attrs = (None, None, None, None, None, None, None)
        self.signal_policies = {attr: dict(cfg) for attr, cfg in SIGNAL_POLICIES.items()}
        self.category_subcategories = defaultdict(set)
        for _, category, subcategory, _, _, _, _ in self.product_attrs.values():
            if not _is_real_source_token(category):
                continue
            if _is_real_source_token(subcategory):
                self.category_subcategories[category].add(subcategory)
        self.product_source_specs = {
            int(pid): self._source_spec(category, subcategory, pid=int(pid))
            for pid, (_, category, subcategory, _, _, _, _) in self.product_attrs.items()
        }

        self.item_price_lookup = self._build_item_price_lookup(products_indexed)
        self.item_price_norm_lookup = _normalize_lookup(self.item_price_lookup)

        self.order_freq, self.order_last_date = self._build_order_tables()
        self.view_freq, self.view_last_date = self._build_view_tables()
        self.order_days_since = self._to_days_since(self.order_last_date)

        self.purchased_dict = self._build_user_interaction_sets(self.order_freq)
        self.viewed_dict = self._build_user_interaction_sets(self.view_freq)
        self.prod_ids_array = np.array([int(pid) for pid in self.prod_ids.tolist()], dtype=np.int64)
        self.substitute_neighbors = self._build_substitute_neighbor_lookup()
        raw_bought_together = ((self.catalog_context.get("relations", {}) or {}).get("bought_together", {}))
        self.bought_together_global = raw_bought_together if isinstance(raw_bought_together, dict) else {}
        self.user_twin_states = {
            int(cid): (
                signals_map.get(int(cid), {})
                if is_user_twin_state(signals_map.get(int(cid), {}))
                else make_empty_user_twin_state(user_id=int(cid), provenance="dataset_init")
            )
            for cid in self.client_ids
        }

        self.branch_feature_schema = {
            "user_state": {
                "intent_summary": list(USER_STATE_INTENT_SUMMARY_FEATURES),
                "ownership_summary": list(USER_STATE_OWNERSHIP_SUMMARY_FEATURES),
                "intent_cluster": list(INTENT_CLUSTER_FEATURE_NAMES),
                "ownership_cluster": list(OWNERSHIP_CLUSTER_FEATURE_NAMES),
            },
            "collaborative": {
                "channel_support": list(COLLABORATIVE_CHANNEL_SUPPORT_FEATURES),
                "cluster": list(COLLABORATIVE_CLUSTER_FEATURE_NAMES),
            },
            "pair_context": {
                "context": list(PAIR_CONTEXT_FEATURE_NAMES),
            },
        }
        self.branch_input_dims = {
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

        self.confirmed_rating_frame, self.global_rating_mean = self._build_confirmed_rating_frame()
        self.rating_lookup = self._build_exact_pair_rating_lookup(self.confirmed_rating_frame)
        raw_samples = self._build_samples(
            num_workers=self.sample_build_workers,
            progress_callback=progress_callback,
        )
        self._store_sample_columns(raw_samples)
        self._release_construction_state()

    def _store_sample_columns(self, samples):
        client_ids = np.asarray(samples.get("client_ids", np.empty((0,), dtype=np.int64)), dtype=np.int64)
        product_ids = np.asarray(samples.get("product_ids", np.empty((0,), dtype=np.int64)), dtype=np.int64)
        label_reg = np.asarray(samples.get("label_reg", np.empty((0,), dtype=np.float32)), dtype=np.float32)
        label_rank = np.asarray(samples.get("label_rank", np.empty((0,), dtype=np.float32)), dtype=np.float32)
        label_conf = np.asarray(samples.get("label_conf", np.empty((0,), dtype=np.float32)), dtype=np.float32)
        feature_metric_arrays = {
            name: np.asarray(samples.get(f"metric_{name}", np.empty((0,), dtype=np.float32)), dtype=np.float32)
            for name in SAMPLE_FEATURE_METRIC_NAMES
        }
        n_samples = int(client_ids.shape[0])
        if not (
            product_ids.shape[0] == n_samples
            and label_reg.shape[0] == n_samples
            and label_rank.shape[0] == n_samples
            and label_conf.shape[0] == n_samples
            and all(arr.shape[0] == n_samples for arr in feature_metric_arrays.values())
        ):
            raise RuntimeError("Sample column lengths are inconsistent.")
        self.num_samples = n_samples
        if n_samples <= 0:
            self.sample_client_ids = np.empty((0,), dtype=np.int64)
            self.sample_product_ids = np.empty((0,), dtype=np.int64)
            self.sample_label_reg = np.empty((0,), dtype=np.float32)
            self.sample_label_rank = np.empty((0,), dtype=np.float32)
            self.sample_label_conf = np.empty((0,), dtype=np.float32)
            self.sample_user_indices = np.empty((0,), dtype=np.int32)
            self.sample_item_indices = np.empty((0,), dtype=np.int32)
            self.sample_feature_metrics = {
                name: np.empty((0,), dtype=np.float32) for name in SAMPLE_FEATURE_METRIC_NAMES
            }
            return

        self.sample_client_ids = client_ids
        self.sample_product_ids = product_ids
        self.sample_label_reg = label_reg
        self.sample_label_rank = label_rank
        self.sample_label_conf = label_conf
        self.sample_user_indices = np.fromiter(
            (int(self.client_index_map.get(int(cid), 0)) for cid in self.sample_client_ids),
            dtype=np.int32,
            count=n_samples,
        )
        self.sample_item_indices = np.fromiter(
            (int(self.prod_index_map.get(int(pid), 0)) for pid in self.sample_product_ids),
            dtype=np.int32,
            count=n_samples,
        )
        order = np.argsort(self.sample_user_indices, kind="stable")
        self.sample_client_ids = self.sample_client_ids[order]
        self.sample_product_ids = self.sample_product_ids[order]
        self.sample_label_reg = self.sample_label_reg[order]
        self.sample_label_rank = self.sample_label_rank[order]
        self.sample_label_conf = self.sample_label_conf[order]
        self.sample_user_indices = self.sample_user_indices[order]
        self.sample_item_indices = self.sample_item_indices[order]
        self.sample_feature_metrics = {
            name: arr[order] for name, arr in feature_metric_arrays.items()
        }

    def _release_construction_state(self):
        for attr in (
            "clients_df",
            "products_df",
            "orders_df",
            "order_items_df",
            "views_df",
            "client_ids",
            "prod_ids",
            "client_index_map",
            "prod_index_map",
            "cat_lookup",
            "order_freq",
            "order_last_date",
            "view_freq",
            "view_last_date",
            "prod_ids_array",
            "confirmed_rating_frame",
        ):
            setattr(self, attr, None)

    def _sample_payload(self, sample_idx):
        idx = int(sample_idx)
        return (
            int(self.sample_client_ids[idx]),
            int(self.sample_product_ids[idx]),
            float(self.sample_label_reg[idx]),
            float(self.sample_label_rank[idx]),
            int(self.sample_user_indices[idx]),
        )

    def _sample_feature_metrics(self, sample_idx):
        idx = int(sample_idx)
        return {
            name: float(self.sample_feature_metrics[name][idx])
            for name in SAMPLE_FEATURE_METRIC_NAMES
        }

    def _build_product_attr_lookups(self):
        raw_lookup = self.catalog_context.get("product_attrs", {}) if isinstance(self.catalog_context, dict) else {}
        if not isinstance(raw_lookup, dict) or not raw_lookup:
            raise RuntimeError("Missing catalog product_attrs in signals; preprocess output is inconsistent with current dataset contract.")

        lookups = {"Category": {}, "Subcategory": {}, "Brand": {}, "Size": {}, "Color": {}, "Material": {}, "Keyword": {}}
        for raw_pid in self.prod_ids:
            pid = int(raw_pid)
            attrs = raw_lookup.get(str(pid), raw_lookup.get(pid, {}))
            if not isinstance(attrs, dict):
                attrs = {}
            lookups["Brand"][pid] = _normalize_attr_token(attrs.get("brand"))
            lookups["Category"][pid] = _normalize_attr_token(attrs.get("category"))
            lookups["Subcategory"][pid] = _normalize_attr_token(attrs.get("subcategory"))
            sizes = _normalize_attr_tokens(attrs.get("size"))
            colors = _normalize_attr_tokens(attrs.get("color"))
            materials = _normalize_attr_tokens(attrs.get("material"))
            keywords = _normalize_attr_tokens(attrs.get("keyword"))
            lookups["Size"][pid] = sizes if sizes else None
            lookups["Color"][pid] = colors if colors else None
            lookups["Material"][pid] = materials if materials else None
            lookups["Keyword"][pid] = keywords if keywords else None
        return lookups

    def _clear_transient_user_runtime_caches(self, state):
        if not isinstance(state, dict):
            return
        for key in (
            "source_feature_cache",
            "intent_cluster_tensor_cache",
            "intent_cluster_array_cache",
            "ownership_cluster_array_cache",
            "collaborative_cluster_array_cache",
            "best_collaborative_signal_cache",
            "cluster_match_cache",
            "global_attr_signal_cache",
            "global_budget_alignment_cache",
        ):
            cache = state.get(key)
            if isinstance(cache, dict):
                cache.clear()

    def _build_substitute_neighbor_lookup(self):
        relations = self.catalog_context.get("relations", {}) if isinstance(self.catalog_context, dict) else {}
        raw_lookup = relations.get("substitute_neighbors", {}) if isinstance(relations, dict) else {}
        if not isinstance(raw_lookup, dict):
            return {}
        out = {}
        for raw_pid, neighbors in raw_lookup.items():
            try:
                pid = int(raw_pid)
            except Exception:
                continue
            if not isinstance(neighbors, dict):
                continue
            normalized = {}
            for raw_rid, raw_strength in neighbors.items():
                try:
                    rid = int(raw_rid)
                except Exception:
                    continue
                strength = _clip(raw_strength, 0.0, 1.0)
                if strength <= 0.0:
                    continue
                normalized[rid] = strength
            if normalized:
                out[pid] = normalized
        return out

    def get_user_twin_state(self, cid):
        user_id = int(cid)
        payload = self.user_twin_states.get(user_id)
        if not is_user_twin_state(payload):
            payload = make_empty_user_twin_state(user_id=user_id, provenance="dataset_runtime")
            self.user_twin_states[user_id] = payload
        return payload

    def _build_item_price_lookup(self, products_indexed):
        price_map = {}

        if not self.order_items_df.empty and {"ProductID", "UnitPrice"}.issubset(self.order_items_df.columns):
            work = self.order_items_df[["ProductID", "UnitPrice"]].copy()
            work["UnitPrice"] = pd.to_numeric(work["UnitPrice"], errors="coerce")
            work = work[work["UnitPrice"] > 0]
            if not work.empty:
                grouped = work.groupby("ProductID")["UnitPrice"].median().to_dict()
                price_map.update({int(k): _safe_float(v, 0.0) for k, v in grouped.items()})

        if "Price" in products_indexed.columns:
            fallback = pd.to_numeric(products_indexed["Price"], errors="coerce")
            for pid, val in fallback.items():
                pid = int(pid)
                if pid not in price_map and pd.notna(val) and _safe_float(val, 0.0) > 0:
                    price_map[pid] = _safe_float(val, 0.0)

        return price_map

    def _build_order_tables(self):
        if self.orders_df.empty or self.order_items_df.empty:
            return {}, {}
        if "OrderID" not in self.orders_df.columns or "OrderID" not in self.order_items_df.columns:
            return {}, {}
        if "ClientID" not in self.orders_df.columns or "ProductID" not in self.order_items_df.columns:
            return {}, {}

        merge_cols = ["OrderID", "ClientID"] + (["OrderDate"] if "OrderDate" in self.orders_df.columns else [])
        merged = self.order_items_df.merge(self.orders_df[merge_cols], on="OrderID", how="inner")
        if merged.empty:
            return {}, {}

        freq = merged.groupby(["ClientID", "ProductID"]).size().to_dict()

        last_date = {}
        if "OrderDate" in merged.columns:
            merged["OrderDate"] = pd.to_datetime(merged["OrderDate"], errors="coerce", utc=True)
            last_date = merged.groupby(["ClientID", "ProductID"])["OrderDate"].max().to_dict()

        return freq, last_date

    def _build_view_tables(self):
        if self.views_df.empty or "ClientID" not in self.views_df.columns or "ProductIDs" not in self.views_df.columns:
            return {}, {}

        work = self.views_df.copy()
        if "ViewDate" in work.columns:
            work["ViewDate"] = pd.to_datetime(work["ViewDate"], errors="coerce", utc=True)
        else:
            work["ViewDate"] = pd.NaT

        work["ProductIDs"] = work["ProductIDs"].apply(_split_product_ids)
        exploded = work[["ClientID", "ViewDate", "ProductIDs"]].explode("ProductIDs")
        exploded = exploded.dropna(subset=["ProductIDs"]).copy()
        if exploded.empty:
            return {}, {}
        exploded["ProductID"] = exploded["ProductIDs"].astype(int)

        freq = exploded.groupby(["ClientID", "ProductID"]).size().to_dict()
        last_date = exploded.groupby(["ClientID", "ProductID"])["ViewDate"].max().to_dict()
        return freq, last_date

    def _to_days_since(self, last_date_map):
        today = to_utc_timestamp()
        out = {}
        for key, dt in last_date_map.items():
            if pd.isna(dt):
                continue
            out[key] = max(int((today - to_utc_timestamp(dt)).days), 0)
        return out

    def _build_user_interaction_sets(self, pair_freq):
        by_client = defaultdict(set)
        for cid, pid in pair_freq.keys():
            by_client[cid].add(int(pid))
        return {cid: by_client.get(cid, set()) for cid in self.client_ids}

    def _build_pair_context_values(self, cid, pid, metrics, state, user_signal):
        return np.asarray(
            [
                _clip(metrics.get("intent_score", 0.0)),
                _clip(metrics.get("source_intent_score", 0.0)),
                _clip(metrics.get("family_local_intent", 0.0)),
                _clip(metrics.get("source_purchase_recency", 0.0)),
                _clip(metrics.get("source_family_recovery", 0.0)),
                _clip(metrics.get("expected_cycle_norm", 0.0)),
                _clip(metrics.get("user_cycle_ratio", 0.0)),
                _clip(metrics.get("user_faster_than_expected", 0.0)),
                _clip(metrics.get("user_slower_than_expected", 0.0)),
                _clip(metrics.get("same_item_penalty", 0.0)),
                _clip(metrics.get("substitute_penalty", 0.0)),
                _clip(metrics.get("cluster_suppression", 0.0)),
                _clip(metrics.get("brand_match", 0.0)),
                _clip(metrics.get("keyword_match", 0.0)),
                _clip(metrics.get("size_match", 0.0)),
                _clip(metrics.get("color_match", 0.0)),
                _clip(metrics.get("material_match", 0.0)),
                _clip(metrics.get("source_purchase_score", 0.0)),
                _clip(metrics.get("source_budget_alignment", 0.0)),
                _clip(metrics.get("global_budget_alignment", 0.0)),
                _clip(metrics.get("price_norm", 0.0)),
                _clip(metrics.get("intent_momentum", 0.0)),
                _clip(metrics.get("interaction_boost", 0.0)),
            ],
            dtype=np.float32,
        )

    def _build_pair_context_tensors(self, cid, pid, metrics, state, user_signal):
        return torch.from_numpy(self._build_pair_context_values(cid, pid, metrics, state, user_signal))

    def _build_pair_context_array(self, cid, pid, metrics, state, user_signal):
        return self._build_pair_context_values(cid, pid, metrics, state, user_signal)

    def _summary_values(self, values, names):
        values = values if isinstance(values, dict) else {}
        return np.asarray([_safe_float(values.get(name, 0.0), 0.0) for name in names], dtype=np.float32)

    def _summary_tensor(self, values, names):
        return torch.from_numpy(self._summary_values(values, names))

    def _summary_array(self, values, names):
        return self._summary_values(values, names)

    def _cluster_signal_match(self, cluster, attr, item_value):
        signals = cluster.get("signals", {}) if isinstance(cluster, dict) else {}
        attr_entry = signals.get(attr, {})
        if not isinstance(attr_entry, dict):
            return 0.0
        score_map = attr_entry.get("scores", {})
        if not isinstance(score_map, dict):
            return 0.0
        if item_value is None:
            return 0.0
        if isinstance(item_value, (tuple, list, set)):
            vals = [self._cluster_signal_match(cluster, attr, v) for v in item_value]
            return max(vals) if vals else 0.0
        return _safe_float(score_map.get(str(item_value), 0.0), 0.0)

    def _cached_cluster_signal_match(self, state, cluster, attr, item_value):
        cluster_key = str(cluster.get("key", "")).strip() if isinstance(cluster, dict) else ""
        value_key = _cacheable_attr_value(item_value)
        cache = state.get("cluster_match_cache", {}) if isinstance(state, dict) else {}
        if cluster_key and value_key is not None and isinstance(cache, dict):
            cache_key = (cluster_key, str(attr), value_key)
            cached = cache.get(cache_key)
            if cached is not None:
                return float(cached)
            score = self._cluster_signal_match(cluster, attr, item_value)
            cache[cache_key] = float(score)
            return float(score)
        return self._cluster_signal_match(cluster, attr, item_value)

    def _cluster_scalar_signal(self, cluster, key):
        signals = cluster.get("signals", {}) if isinstance(cluster, dict) else {}
        value = signals.get(key, 0.0)
        if isinstance(value, dict):
            return 0.0
        return _clip(value)

    def _cluster_budget_alignment(self, cluster, item_price):
        signals = cluster.get("signals", {}) if isinstance(cluster, dict) else {}
        target = signals.get("budget", np.nan)
        if isinstance(target, dict):
            return 0.0
        score = self._budget_alignment_from_target(target, item_price)
        return 0.0 if score is None else _clip(score)

    def _cluster_budget_features(self, cluster, item_price):
        signals = cluster.get("signals", {}) if isinstance(cluster, dict) else {}
        target = signals.get("budget", np.nan)
        if isinstance(target, dict):
            target = np.nan
        target = _safe_float(target, np.nan)
        price = _safe_float(item_price, np.nan)
        if not np.isfinite(target) or target <= 0.0 or not np.isfinite(price) or price <= 0.0:
            return {
                "target_log": 0.0,
                "price_log": 0.0,
                "ratio": 1.0,
                "delta_ratio": 0.0,
                "over": 0.0,
                "under": 0.0,
                "alignment": 0.0,
            }
        ratio = price / max(target, 1e-6)
        delta_ratio = (price - target) / max(target, 1e-6)
        over = max(delta_ratio, 0.0)
        under = max(-delta_ratio, 0.0)
        return {
            "target_log": float(np.log1p(target)),
            "price_log": float(np.log1p(price)),
            "ratio": float(ratio),
            "delta_ratio": float(delta_ratio),
            "over": float(over),
            "under": float(under),
            "alignment": self._cluster_budget_alignment(cluster, item_price),
        }

    def _pad_feature_rows(self, rows, feat_dim, limit=None):
        row_limit = int(limit if limit is not None else self.max_seq_len)
        padded = np.zeros((row_limit, feat_dim), dtype=np.float32)
        if not rows:
            return padded
        clipped_rows = rows[:row_limit]
        padded[:len(clipped_rows)] = np.asarray(clipped_rows, dtype=np.float32)
        return padded

    def _build_intent_cluster_sequence_values(self, intent_clusters, ownership_clusters, category, subcategory):
        item_cluster_key = self._source_spec(category, subcategory).get("key")
        ownership_by_key = {
            str(cluster.get("key", "")): cluster
            for cluster in (ownership_clusters or [])
            if isinstance(cluster, dict)
        }
        rows = []
        for cluster in intent_clusters or []:
            cluster_key = str(cluster.get("key", ""))
            active_cluster = 1.0 if cluster_key == item_cluster_key else 0.0
            fulfilled_cluster = ownership_by_key.get(cluster_key, {})
            rows.append([
                _clip(self._cluster_scalar_signal(cluster, "source_strength")),
                _clip(self._cluster_scalar_signal(cluster, "view_engagement")),
                _clip(self._cluster_scalar_signal(cluster, "adjacent_engagement")),
                active_cluster,
                _clip(self._cluster_scalar_signal(fulfilled_cluster, "source_strength")),
            ])
        return self._pad_feature_rows(rows, len(INTENT_CLUSTER_FEATURE_NAMES))

    def _build_intent_cluster_sequence_tensor(self, intent_clusters, ownership_clusters, category, subcategory):
        return torch.from_numpy(
            self._build_intent_cluster_sequence_values(intent_clusters, ownership_clusters, category, subcategory)
        )

    def _build_ownership_cluster_sequence_values(self, ownership_clusters, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        item_cluster_key = self._source_spec(category, subcategory).get("key")
        rows = []
        for cluster in ownership_clusters or []:
            active_cluster = 1.0 if cluster.get("key") == item_cluster_key else 0.0
            signals = cluster.get("signals", {}) if isinstance(cluster, dict) else {}
            budget_target = _safe_float(signals.get("budget", np.nan), np.nan)
            rows.append([
                _clip(self._cluster_scalar_signal(cluster, "source_strength")),
                active_cluster,
                float(np.log1p(budget_target)) if np.isfinite(budget_target) and budget_target > 0.0 else 0.0,
            ])
        return self._pad_feature_rows(rows, len(OWNERSHIP_CLUSTER_FEATURE_NAMES))

    def _build_ownership_cluster_sequence_tensor(self, ownership_clusters, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        return torch.from_numpy(
            self._build_ownership_cluster_sequence_values(
                ownership_clusters,
                pid,
                category,
                subcategory,
                brand,
                size,
                color,
                material,
                keyword,
                state=state,
            )
        )

    def _build_ownership_cluster_sequence_array(self, ownership_clusters, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        return self._build_ownership_cluster_sequence_values(
            ownership_clusters,
            pid,
            category,
            subcategory,
            brand,
            size,
            color,
            material,
            keyword,
            state=state,
        )

    def _build_intent_cluster_sequence_array(self, intent_clusters, ownership_clusters, category, subcategory):
        return self._build_intent_cluster_sequence_values(intent_clusters, ownership_clusters, category, subcategory)

    def _build_collaborative_cluster_sequence_values(self, clusters, channel, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        item_cluster_key = self._source_spec(category, subcategory).get("key")
        channel_flags = {
            "age": (1.0, 0.0, 0.0),
            "gender": (0.0, 1.0, 0.0),
            "location": (0.0, 0.0, 1.0),
        }.get(channel, (0.0, 0.0, 0.0))
        support_score = 0.0
        if isinstance(state, dict):
            support_score = _clip((state.get("coll_channel_support_scores", {}) or {}).get(channel, 0.0))
        rows = []
        for cluster in clusters or []:
            active_cluster = 1.0 if cluster.get("key") == item_cluster_key else 0.0
            rows.append([
                channel_flags[0],
                channel_flags[1],
                channel_flags[2],
                support_score,
                self._cluster_scalar_signal(cluster, "source_strength"),
                active_cluster,
            ])
        return self._pad_feature_rows(rows, len(COLLABORATIVE_CLUSTER_FEATURE_NAMES), limit=self.collab_top_k)

    def _build_collaborative_cluster_sequence_tensor(self, clusters, channel, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        return torch.from_numpy(
            self._build_collaborative_cluster_sequence_values(
                clusters,
                channel,
                pid,
                category,
                subcategory,
                brand,
                size,
                color,
                material,
                keyword,
                state=state,
            )
        )

    def _build_collaborative_cluster_sequence_array(self, clusters, channel, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        return self._build_collaborative_cluster_sequence_values(
            clusters,
            channel,
            pid,
            category,
            subcategory,
            brand,
            size,
            color,
            material,
            keyword,
            state=state,
        )

    def _build_flat_collaborative_clusters_values(self, collaborative, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        per_channel = []
        channels = collaborative.get("channels", {}) if isinstance(collaborative.get("channels", {}), dict) else {}
        for channel in COLLAB_BASE_CHANNELS:
            per_channel.append(
                self._build_collaborative_cluster_sequence_values(
                    channels.get(channel, {}).get("clusters", []),
                    channel,
                    pid,
                    category,
                    subcategory,
                    brand,
                    size,
                    color,
                    material,
                    keyword,
                    state=state,
                )
            )
        if not per_channel:
            return np.zeros(
                (self.collab_top_k * len(COLLAB_BASE_CHANNELS), len(COLLABORATIVE_CLUSTER_FEATURE_NAMES)),
                dtype=np.float32,
            )
        return np.concatenate(per_channel, axis=0)

    def _build_flat_collaborative_clusters(self, collaborative, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        return torch.from_numpy(
            self._build_flat_collaborative_clusters_values(
                collaborative,
                pid,
                category,
                subcategory,
                brand,
                size,
                color,
                material,
                keyword,
                state=state,
            )
        )

    def _build_flat_collaborative_clusters_array(self, collaborative, pid, category, subcategory, brand, size, color, material, keyword, state=None):
        return self._build_flat_collaborative_clusters_values(
            collaborative,
            pid,
            category,
            subcategory,
            brand,
            size,
            color,
            material,
            keyword,
            state=state,
        )

    def _build_collaborative_channel_support_values(self, collaborative):
        values = []
        channels = collaborative.get("channels", {}) if isinstance(collaborative.get("channels", {}), dict) else {}
        for channel in COLLAB_BASE_CHANNELS:
            channel_block = channels.get(channel, {}) if isinstance(channels.get(channel, {}), dict) else {}
            channel_summary = channel_block.get("summary", {}) if isinstance(channel_block.get("summary", {}), dict) else {}
            channel_support = channel_summary
            for feature in COLLAB_CHANNEL_SUPPORT_FEATURES:
                values.append(_safe_float(channel_support.get(feature, 0.0), 0.0))
        return np.asarray(values, dtype=np.float32)

    def _build_collaborative_channel_support_tensor(self, collaborative):
        return torch.from_numpy(self._build_collaborative_channel_support_values(collaborative))

    def _build_collaborative_channel_support_array(self, collaborative):
        return self._build_collaborative_channel_support_values(collaborative)

    def _branch_bundle_context(self, pid, state):
        brand, category, subcategory, size, color, material, keyword = self.product_attrs.get(pid, self.empty_attrs)
        source_cache_key = self._source_spec(category, subcategory).get("key") or f"{category}::{subcategory}"
        branches = state.get("branches", {}) if isinstance(state.get("branches", {}), dict) else {}
        user_state = branches.get("user_state", {}) if isinstance(branches.get("user_state", {}), dict) else {}
        collaborative = branches.get("collaborative", {}) if isinstance(branches.get("collaborative", {}), dict) else {}
        intent_state = user_state.get("intent", {}) if isinstance(user_state.get("intent", {}), dict) else {}
        ownership_state = user_state.get("ownership", {}) if isinstance(user_state.get("ownership", {}), dict) else {}
        return {
            "brand": brand,
            "category": category,
            "subcategory": subcategory,
            "size": size,
            "color": color,
            "material": material,
            "keyword": keyword,
            "source_cache_key": source_cache_key,
            "user_state": user_state,
            "collaborative": collaborative,
            "intent_summary": intent_state.get("summary", {}) if isinstance(intent_state.get("summary", {}), dict) else {},
            "ownership_summary": ownership_state.get("summary", {}) if isinstance(ownership_state.get("summary", {}), dict) else {},
        }

    def _build_structured_branch_bundle(self, pid, state, metrics):
        context = self._branch_bundle_context(pid, state)
        intent_clusters = context["user_state"].get("intent", {}).get("clusters", [])
        ownership_clusters = context["user_state"].get("ownership", {}).get("clusters", [])
        intent_summary_values = context["intent_summary"]
        ownership_summary_values = context["ownership_summary"]

        intent_cluster_tensor_cache = state.get("intent_cluster_tensor_cache", {})
        intent_cluster_tensor = (
            intent_cluster_tensor_cache.get(context["source_cache_key"])
            if isinstance(intent_cluster_tensor_cache, dict)
            else None
        )
        if intent_cluster_tensor is None:
            intent_cluster_tensor = self._build_intent_cluster_sequence_tensor(
                intent_clusters,
                ownership_clusters,
                context["category"],
                context["subcategory"],
            )
            if isinstance(intent_cluster_tensor_cache, dict):
                intent_cluster_tensor_cache[context["source_cache_key"]] = intent_cluster_tensor

        user_state_bundle = {
            "intent_summary": self._summary_tensor(intent_summary_values, USER_STATE_INTENT_SUMMARY_FEATURES),
            "intent_clusters": intent_cluster_tensor,
            "ownership_summary": self._summary_tensor(ownership_summary_values, USER_STATE_OWNERSHIP_SUMMARY_FEATURES),
            "ownership_clusters": self._build_ownership_cluster_sequence_tensor(
                ownership_clusters,
                pid,
                context["category"],
                context["subcategory"],
                context["brand"],
                context["size"],
                context["color"],
                context["material"],
                context["keyword"],
                state=state,
            ),
        }
        collaborative_bundle = {
            "channel_support": (
                state.get("collaborative_channel_support_tensor")
                if isinstance(state.get("collaborative_channel_support_tensor"), torch.Tensor)
                else self._build_collaborative_channel_support_tensor(context["collaborative"])
            ),
            "clusters": self._build_flat_collaborative_clusters(
                context["collaborative"],
                pid,
                context["category"],
                context["subcategory"],
                context["brand"],
                context["size"],
                context["color"],
                context["material"],
                context["keyword"],
                state=state,
            ),
        }
        return user_state_bundle, collaborative_bundle

    def _build_structured_branch_bundle_arrays(self, pid, state, metrics):
        context = self._branch_bundle_context(pid, state)
        intent_clusters = context["user_state"].get("intent", {}).get("clusters", [])
        ownership_clusters = context["user_state"].get("ownership", {}).get("clusters", [])
        intent_summary_values = context["intent_summary"]
        ownership_summary_values = context["ownership_summary"]

        intent_cluster_array_cache = state.get("intent_cluster_array_cache", {})
        intent_cluster_array = (
            intent_cluster_array_cache.get(context["source_cache_key"])
            if isinstance(intent_cluster_array_cache, dict)
            else None
        )
        if intent_cluster_array is None:
            intent_cluster_array = self._build_intent_cluster_sequence_array(
                intent_clusters,
                ownership_clusters,
                context["category"],
                context["subcategory"],
            )
            if isinstance(intent_cluster_array_cache, dict):
                intent_cluster_array_cache[context["source_cache_key"]] = intent_cluster_array

        user_state_bundle = {
            "intent_summary": (
                state.get("intent_summary_array")
                if isinstance(state.get("intent_summary_array"), np.ndarray)
                else self._summary_array(intent_summary_values, USER_STATE_INTENT_SUMMARY_FEATURES)
            ),
            "intent_clusters": intent_cluster_array,
            "ownership_summary": (
                state.get("ownership_summary_array")
                if isinstance(state.get("ownership_summary_array"), np.ndarray)
                else self._summary_array(ownership_summary_values, USER_STATE_OWNERSHIP_SUMMARY_FEATURES)
            ),
        }
        ownership_cluster_array_cache = state.get("ownership_cluster_array_cache", {})
        ownership_cluster_array = (
            ownership_cluster_array_cache.get(context["source_cache_key"])
            if isinstance(ownership_cluster_array_cache, dict)
            else None
        )
        if ownership_cluster_array is None:
            ownership_cluster_array = self._build_ownership_cluster_sequence_array(
                ownership_clusters,
                pid,
                context["category"],
                context["subcategory"],
                context["brand"],
                context["size"],
                context["color"],
                context["material"],
                context["keyword"],
                state=state,
            )
            if isinstance(ownership_cluster_array_cache, dict):
                ownership_cluster_array_cache[context["source_cache_key"]] = ownership_cluster_array
        user_state_bundle["ownership_clusters"] = ownership_cluster_array
        collaborative_bundle = {
            "channel_support": (
                state.get("collaborative_channel_support_array")
                if isinstance(state.get("collaborative_channel_support_array"), np.ndarray)
                else self._build_collaborative_channel_support_array(context["collaborative"])
            ),
        }
        collaborative_cluster_array_cache = state.get("collaborative_cluster_array_cache", {})
        collaborative_cluster_array = (
            collaborative_cluster_array_cache.get(context["source_cache_key"])
            if isinstance(collaborative_cluster_array_cache, dict)
            else None
        )
        if collaborative_cluster_array is None:
            collaborative_cluster_array = self._build_flat_collaborative_clusters_array(
                context["collaborative"],
                pid,
                context["category"],
                context["subcategory"],
                context["brand"],
                context["size"],
                context["color"],
                context["material"],
                context["keyword"],
                state=state,
            )
            if isinstance(collaborative_cluster_array_cache, dict):
                collaborative_cluster_array_cache[context["source_cache_key"]] = collaborative_cluster_array
        collaborative_bundle["clusters"] = collaborative_cluster_array
        return user_state_bundle, collaborative_bundle

    def _get_user_pair_state(self, cid, user_signal=None):
        if user_signal is None:
            user_signal = self.get_user_twin_state(cid)
        return self._prepare_user_pair_state(user_signal)

    def get_feature_bundle(self, cid, pid, *, user_signal=None, state=None, metrics=None):
        if user_signal is None:
            user_signal = self.get_user_twin_state(cid)
        if state is None:
            state = self._get_user_pair_state(cid, user_signal=user_signal)
        if metrics is None:
            viewed = self.viewed_dict.get(cid, set())
            metrics = self._compute_pair_metrics(
                cid=cid,
                pid=pid,
                state=state,
                viewed=viewed,
            )
        user_state, collaborative = self._build_structured_branch_bundle(
            pid=pid,
            state=state,
            metrics=metrics,
        )
        pair_context = self._build_pair_context_tensors(
            cid=cid,
            pid=pid,
            metrics=metrics,
            state=state,
            user_signal=user_signal,
        )
        return user_state, collaborative, pair_context

    def get_feature_bundle_arrays(self, cid, pid, *, user_signal=None, state=None, metrics=None):
        if user_signal is None:
            user_signal = self.get_user_twin_state(cid)
        if state is None:
            state = self._get_user_pair_state(cid, user_signal=user_signal)
        if metrics is None:
            viewed = self.viewed_dict.get(cid, set())
            metrics = self._compute_pair_metrics(
                cid=cid,
                pid=pid,
                state=state,
                viewed=viewed,
            )
        user_state, collaborative = self._build_structured_branch_bundle_arrays(
            pid=pid,
            state=state,
            metrics=metrics,
        )
        pair_context = self._build_pair_context_array(
            cid=cid,
            pid=pid,
            metrics=metrics,
            state=state,
            user_signal=user_signal,
        )
        return user_state, collaborative, pair_context

    def _best_collaborative_signal(self, state, pid):
        _, category, subcategory, _, _, _, _ = self.product_attrs.get(pid, self.empty_attrs)
        source_spec = self._source_spec_for_pid(pid, category, subcategory)
        source_cache_key = source_spec.get("key") or f"{category}::{subcategory}"
        score_cache = state.get("best_collaborative_signal_cache", {})
        if isinstance(score_cache, dict) and source_cache_key in score_cache:
            return float(score_cache[source_cache_key])
        branches = state.get("branches", {}) if isinstance(state.get("branches", {}), dict) else {}
        collaborative = branches.get("collaborative", {}) if isinstance(branches.get("collaborative", {}), dict) else {}
        channels = collaborative.get("channels", {}) if isinstance(collaborative.get("channels", {}), dict) else {}
        cluster_maps = state.get("coll_cluster_maps", {}) if isinstance(state.get("coll_cluster_maps", {}), dict) else {}
        channel_support_scores = state.get("coll_channel_support_scores", {}) if isinstance(state, dict) else {}

        best_score = 0.0
        for channel in COLLAB_BASE_CHANNELS:
            cluster = self._select_cluster(cluster_maps.get(channel, {}), category, subcategory, scope="source_local", pid=pid)
            cluster_score = _clip(self._cluster_scalar_signal(cluster, "source_strength"))
            support_score = _clip(channel_support_scores.get(channel, 0.0))
            score = cluster_score if cluster_score > 0.0 else support_score
            if score > best_score:
                best_score = score
        if isinstance(score_cache, dict):
            score_cache[source_cache_key] = float(best_score)
        return float(best_score)

    def _build_confirmed_rating_frame(self):
        if self.order_items_df.empty or self.orders_df.empty:
            return pd.DataFrame(columns=["ClientID", "ProductID", "Rating"]), float(RATING_FALLBACK_VALUE)
        if "Rating" not in self.order_items_df.columns:
            return pd.DataFrame(columns=["ClientID", "ProductID", "Rating"]), float(RATING_FALLBACK_VALUE)
        if "OrderID" not in self.order_items_df.columns or "OrderID" not in self.orders_df.columns:
            return pd.DataFrame(columns=["ClientID", "ProductID", "Rating"]), float(RATING_FALLBACK_VALUE)
        if "ClientID" not in self.orders_df.columns or "ProductID" not in self.order_items_df.columns:
            return pd.DataFrame(columns=["ClientID", "ProductID", "Rating"]), float(RATING_FALLBACK_VALUE)

        merged = self.order_items_df.merge(self.orders_df[["OrderID", "ClientID"]], on="OrderID", how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["ClientID", "ProductID", "Rating"]), float(RATING_FALLBACK_VALUE)
        merged["Rating"] = pd.to_numeric(merged["Rating"], errors="coerce")
        rated = merged.dropna(subset=["Rating", "ClientID", "ProductID"]).copy()
        if rated.empty:
            return pd.DataFrame(columns=["ClientID", "ProductID", "Rating"]), float(RATING_FALLBACK_VALUE)
        grouped = (
            rated.groupby(["ClientID", "ProductID"], as_index=False)["Rating"]
            .mean()
        )
        grouped["Rating"] = grouped["Rating"].clip(1.0, 5.0)
        global_rating_mean = float(np.clip(grouped["Rating"].mean(), 1.0, 5.0))
        return grouped, global_rating_mean

    def _build_exact_pair_rating_lookup(self, confirmed_ratings):
        if confirmed_ratings is None or confirmed_ratings.empty:
            return {}
        indexed = confirmed_ratings.set_index(["ClientID", "ProductID"])["Rating"]
        return indexed.to_dict()

    def _profile_rank_estimate(self, cid, pid, state=None, viewed=None, pair_metrics=None):
        if pair_metrics is None:
            if state is None:
                user_signal = self.get_user_twin_state(cid)
                state = self._get_user_pair_state(cid, user_signal=user_signal)
            if viewed is None:
                viewed = self.viewed_dict.get(cid, set())
            pair_metrics = self._compute_pair_metrics(
                cid=cid,
                pid=pid,
                state=state,
                viewed=viewed,
            )

        source_purchase = _clip(pair_metrics.get("source_purchase_score", 0.0))
        source_recovery = _clip(pair_metrics.get("source_family_recovery", 0.0))
        source_budget_alignment = _clip(pair_metrics.get("source_budget_alignment", 0.5))
        global_budget_alignment = _clip(pair_metrics.get("global_budget_alignment", 0.5))
        same_item_penalty = _clip(pair_metrics.get("same_item_penalty", 0.0))
        substitute_penalty = _clip(pair_metrics.get("substitute_penalty", 0.0))
        cluster_suppression = _clip(pair_metrics.get("cluster_suppression", 0.0))
        product_fit = _clip(
            0.18 * _clip(pair_metrics.get("brand_match", 0.0))
            + 0.18 * _clip(pair_metrics.get("keyword_match", 0.0))
            + 0.12 * _clip(pair_metrics.get("size_match", 0.0))
            + 0.08 * _clip(pair_metrics.get("color_match", 0.0))
            + 0.10 * _clip(pair_metrics.get("material_match", 0.0))
            + 0.16 * source_purchase
            + 0.12 * source_budget_alignment
            + 0.06 * global_budget_alignment
        )
        family_context = _clip(
            0.28 * _clip(pair_metrics.get("intent_score", 0.0))
            + 0.12 * _clip(pair_metrics.get("source_intent_score", 0.0))
            + 0.18 * _clip(pair_metrics.get("intent_momentum", 0.0))
            + 0.22 * source_recovery
            + 0.20 * _clip(pair_metrics.get("source_purchase_recency", 0.0))
        )
        suppression = _clip(max(same_item_penalty, substitute_penalty, cluster_suppression))
        coll_score = _clip(self._best_collaborative_signal(state=state, pid=pid))
        ownership_strength = _clip(max(product_fit, source_purchase, source_recovery))
        collaborative_weight = 0.08 + 0.22 * (1.0 - ownership_strength)
        rank_signal = _clip(
            0.34 * product_fit
            + 0.22 * family_context
            + 0.18 * source_purchase
            + 0.12 * source_recovery
            + 0.08 * (1.0 - suppression)
            + collaborative_weight * coll_score
        )
        profile_rank = float(np.clip(1.0 + 4.0 * rank_signal, 1.0, 5.0))
        profile_strength = _clip(
            0.38 * product_fit
            + 0.18 * family_context
            + 0.14 * source_purchase
            + 0.10 * source_recovery
            + (0.05 + 0.20 * (1.0 - ownership_strength)) * coll_score
            + 0.10 * (1.0 - suppression)
        )
        return profile_rank, float(profile_strength)

    def _label_rank(self, cid, pid, state=None, viewed=None, pair_metrics=None):
        rating = self.rating_lookup.get((cid, pid))
        if rating is not None and not pd.isna(rating):
            return float(np.clip(_safe_float(rating, self.global_rating_mean), 1.0, 5.0))

        profile_rank, _profile_strength = self._profile_rank_estimate(
            cid=cid,
            pid=pid,
            state=state,
            viewed=viewed,
            pair_metrics=pair_metrics,
        )
        return float(np.clip(profile_rank, 1.0, 5.0))

    def _recency_days(self, recency_map, attr_value):
        if not isinstance(recency_map, dict):
            return None
        if attr_value not in recency_map:
            return None
        days = _safe_float(recency_map.get(attr_value, np.nan), np.nan)
        if not np.isfinite(days):
            return None
        return max(days, 0.0)

    def _source_recency_days(self, state, category, subcategory, pid=None):
        source = self._source_spec(category, subcategory, pid=pid)
        if source["level"] == "family" and source.get("value") is not None:
            days = self._recency_days(state.get("recency_by_family", {}), source["value"])
            if days is not None:
                return days
            leaf_key = source.get("parent")
            if leaf_key is not None:
                days = self._recency_days(state.get("recency_by_taxonomy_leaf", {}), leaf_key)
                if days is not None:
                    return days
        if source["level"] == "subcategory" and source.get("value") is not None:
            days = self._recency_days(state.get("recency_by_subcategory", {}), source["value"])
            if days is not None:
                return days
        category_value = source.get("parent") if source["level"] == "subcategory" else source.get("value")
        if category_value is None:
            return None
        return self._recency_days(state.get("recency_by_category", {}), category_value)

    def _source_purchase_recency(self, state, category, subcategory, pid=None):
        source_days = self._source_recency_days(state, category, subcategory, pid=pid)
        if source_days is None:
            return 0.0
        recency_tau = max(21.0, float(self.max_seq_len) * 4.0)
        return _clip(_exp_decay(source_days, recency_tau))

    def _expected_cycle_profile(self, category, subcategory):
        profiles = self.catalog_context.get("replenishment_profiles", {}) if isinstance(self.catalog_context, dict) else {}
        sub_profiles = profiles.get("subcategory", {}) if isinstance(profiles.get("subcategory", {}), dict) else {}
        cat_profiles = profiles.get("category", {}) if isinstance(profiles.get("category", {}), dict) else {}
        sub_token = _normalize_attr_token(subcategory)
        cat_token = _normalize_attr_token(category)
        if _is_real_source_token(sub_token) and isinstance(sub_profiles.get(sub_token), dict):
            return sub_profiles.get(sub_token)
        if _is_real_source_token(cat_token) and isinstance(cat_profiles.get(cat_token), dict):
            return cat_profiles.get(cat_token)
        return {}

    def _cycle_delta_value(self, state, key_prefix, category, subcategory, pid=None, default=0.0):
        source = self._source_spec(category, subcategory, pid=pid)
        if source["level"] == "family" and source.get("value") is not None:
            family_map = state.get(f"{key_prefix}_by_family", {})
            if isinstance(family_map, dict) and source["value"] in family_map:
                return _safe_float(family_map.get(source["value"]), default)
            leaf_key = source.get("parent")
            leaf_map = state.get(f"{key_prefix}_by_taxonomy_leaf", {})
            if leaf_key is not None and isinstance(leaf_map, dict) and leaf_key in leaf_map:
                return _safe_float(leaf_map.get(leaf_key), default)
        sub_map = state.get(f"{key_prefix}_by_subcategory", {})
        cat_map = state.get(f"{key_prefix}_by_category", {})
        sub_token = _normalize_attr_token(subcategory)
        cat_token = _normalize_attr_token(category)
        if _is_real_source_token(sub_token) and isinstance(sub_map, dict) and sub_token in sub_map:
            return _safe_float(sub_map.get(sub_token), default)
        if _is_real_source_token(cat_token) and isinstance(cat_map, dict) and cat_token in cat_map:
            return _safe_float(cat_map.get(cat_token), default)
        return _safe_float(default, 0.0)

    def _post_purchase_phase(self, state):
        days = _safe_float(state.get("days_since_last_purchase", np.nan), np.nan)
        if not np.isfinite(days) or days < 0.0:
            return 0.0
        interval_days = float(np.clip(_safe_float(state.get("observed_cycle_days", 90.0), 90.0), 14.0, 365.0))
        tau = float(np.clip(0.55 * interval_days, 14.0, 120.0))
        return _clip(_exp_decay(days, tau))

    def _source_spec(self, category, subcategory, pid=None):
        if pid is not None:
            family_id = self.product_family_lookup.get(int(pid))
            if family_id is not None:
                meta = self.family_meta_lookup.get(str(family_id), {})
                leaf_key = meta.get("taxonomy_leaf_key", self.product_taxonomy_leaf_lookup.get(int(pid)))
                return {
                    "level": "family",
                    "value": str(family_id),
                    "parent": str(leaf_key) if leaf_key is not None else None,
                    "key": f"family::{family_id}",
                    "category": meta.get("category", _normalize_attr_token(category)),
                    "subcategory": meta.get("subcategory", _normalize_attr_token(subcategory)),
                }
        category_token = _normalize_attr_token(category)
        subcategory_token = _normalize_attr_token(subcategory)
        has_category = _is_real_source_token(category_token)
        has_subcategory = _is_real_source_token(subcategory_token)
        if has_category and has_subcategory and self.category_subcategories.get(category_token):
            return {
                "level": "subcategory",
                "value": subcategory_token,
                "parent": category_token,
                "key": f"subcategory::{subcategory_token}",
            }
        return {
            "level": "category",
            "value": category_token if has_category else None,
            "parent": None,
            "key": f"category::{category_token}" if has_category else None,
        }

    def _source_spec_for_pid(self, pid, category=None, subcategory=None):
        cached = self.product_source_specs.get(int(pid))
        if isinstance(cached, dict):
            return cached
        return self._source_spec(category, subcategory, pid=pid)

    def _cluster_key_map(self, clusters):
        out = {}
        for cluster in clusters or []:
            if not isinstance(cluster, dict):
                continue
            key = str(cluster.get("key", "")).strip()
            if key:
                out[key] = cluster
        return out

    def _select_cluster(self, clusters, category, subcategory, scope="source_local", pid=None):
        cluster_map = clusters if isinstance(clusters, dict) else self._cluster_key_map(clusters)
        if not isinstance(cluster_map, dict) or not cluster_map:
            return None

        source_spec = self._source_spec(category, subcategory, pid=pid)
        keys = []
        if scope == "parent_category":
            category_token = source_spec.get("parent") if source_spec["level"] in {"family", "subcategory"} else source_spec.get("value")
            if category_token:
                keys.append(str(category_token))
            if source_spec.get("key"):
                keys.append(str(source_spec["key"]))
        else:
            if source_spec.get("key"):
                keys.append(str(source_spec["key"]))
            if source_spec["level"] == "family" and source_spec.get("parent"):
                keys.append(str(source_spec["parent"]))
            elif source_spec["level"] == "subcategory" and source_spec.get("parent"):
                keys.append(f"category::{source_spec['parent']}")

        for key in keys:
            cluster = cluster_map.get(str(key))
            if isinstance(cluster, dict):
                return cluster
        return None

    def _aggregate_cluster_attr_signal(self, clusters, attr, value):
        if value is None:
            return 0.0
        score_sum = 0.0
        weight_sum = 0.0
        found = False
        for cluster in clusters or []:
            if not isinstance(cluster, dict):
                continue
            score = self._cluster_signal_match(cluster, attr, value)
            strength = max(0.15, _clip(self._cluster_scalar_signal(cluster, "source_strength")))
            if abs(score) > 1e-8:
                found = True
            score_sum += strength * score
            weight_sum += strength
        if not found or weight_sum <= 1e-8:
            return 0.0
        return float(score_sum / weight_sum)

    def _cached_aggregate_cluster_attr_signal(self, state, clusters, attr, value):
        value_key = _cacheable_attr_value(value)
        cache = state.get("global_attr_signal_cache", {}) if isinstance(state, dict) else {}
        if value_key is not None and isinstance(cache, dict):
            cache_key = (str(attr), value_key)
            cached = cache.get(cache_key)
            if cached is not None:
                return float(cached)
        if value is None:
            score = 0.0
        else:
            score_sum = 0.0
            weight_sum = 0.0
            found = False
            for cluster in clusters or []:
                if not isinstance(cluster, dict):
                    continue
                match_score = self._cached_cluster_signal_match(state, cluster, attr, value)
                strength = max(0.15, _clip(self._cluster_scalar_signal(cluster, "source_strength")))
                if abs(match_score) > 1e-8:
                    found = True
                score_sum += strength * match_score
                weight_sum += strength
            if not found or weight_sum <= 1e-8:
                score = 0.0
            else:
                score = float(score_sum / weight_sum)
        if value_key is not None and isinstance(cache, dict):
            cache[(str(attr), value_key)] = float(score)
        return float(score)

    def _substitute_neighbor_decay(self, cid, pid, price_norm):
        neighbors = self.substitute_neighbors.get(int(pid), {})
        if not isinstance(neighbors, dict) or not neighbors:
            return 0.0

        tau = 35.0 + 110.0 * price_norm
        best = 0.0
        for rid, strength in neighbors.items():
            days = self.order_days_since.get((cid, int(rid)))
            if days is None:
                continue
            decay = _clip(strength, 0.0, 1.0) * _exp_decay(days, tau)
            if decay > best:
                best = decay
        return float(best)

    def _budget_alignment_from_target(self, target_price, item_price):
        target = _safe_float(target_price, np.nan)
        price = _safe_float(item_price, np.nan)
        if not np.isfinite(target) or target <= 0.0 or not np.isfinite(price) or price <= 0.0:
            return None
        if price > target:
            diff = (price - target) / (target + 1e-6)
            return float(np.exp(-1.25 * diff))
        diff = (target - price) / (target + 1e-6)
        return float(np.exp(-0.65 * diff))

    def _global_budget_alignment(self, budgets, item_price):
        if not isinstance(budgets, dict):
            return 0.5
        target = budgets.get("global", np.nan)
        score = self._budget_alignment_from_target(target, item_price)
        return 0.5 if score is None else _clip(score)

    def _source_budget_alignment(self, budgets, category, subcategory, item_price, pid=None):
        if not isinstance(budgets, dict):
            return None
        source = self._source_spec(category, subcategory, pid=pid)
        if source["level"] == "family" and source.get("value") is not None:
            family_map = budgets.get("family", {})
            if isinstance(family_map, dict):
                score = self._budget_alignment_from_target(family_map.get(source["value"], np.nan), item_price)
                if score is not None:
                    return _clip(score)
            leaf_map = budgets.get("taxonomy_leaf", {})
            if isinstance(leaf_map, dict) and source.get("parent") in leaf_map:
                score = self._budget_alignment_from_target(leaf_map.get(source.get("parent"), np.nan), item_price)
                if score is not None:
                    return _clip(score)
        candidates = []
        if source["level"] == "subcategory" and source.get("value") is not None:
            candidates.append(("subcategory", source["value"]))
        if source["level"] == "subcategory" and source.get("parent") is not None:
            candidates.append(("category", source["parent"]))
        elif source["level"] == "category" and source.get("value") is not None:
            candidates.append(("category", source["value"]))

        for key, attr_value in candidates:
            bmap = budgets.get(key, {})
            if not isinstance(bmap, dict) or attr_value not in bmap:
                continue
            score = self._budget_alignment_from_target(bmap.get(attr_value, np.nan), item_price)
            if score is not None:
                return _clip(score)
        return None

    def _prepare_user_pair_state(self, user_signal):
        user_signal = user_signal if isinstance(user_signal, dict) else {}

        def _normalize_branch_clusters(blocks):
            out = []
            if not isinstance(blocks, dict):
                return out
            for cluster_id, raw_block in blocks.items():
                if not isinstance(raw_block, dict):
                    continue
                raw_signals = raw_block.get("signals", {})
                if not isinstance(raw_signals, dict):
                    raw_signals = {}
                signals = {}
                for key, value in raw_signals.items():
                    if isinstance(value, dict):
                        raw_scores = value.get("scores", {})
                        if not isinstance(raw_scores, dict):
                            raw_scores = {}
                        signals[str(key)] = {
                            "scores": {str(k): _safe_float(v, 0.0) for k, v in raw_scores.items()},
                        }
                    else:
                        signals[str(key)] = _safe_float(value, 0.0)
                out.append({
                    "key": str(raw_block.get("key", cluster_id)),
                    "level": raw_block.get("level"),
                    "value": raw_block.get("value"),
                    "parent": raw_block.get("parent"),
                    "category": raw_block.get("category"),
                    "subcategory": raw_block.get("subcategory"),
                    "signals": signals,
                })
            out.sort(
                key=lambda row: (
                    -_clip(row.get("signals", {}).get("source_strength", 0.0)),
                    str(row.get("key", "")),
                )
            )
            return out

        def _normalize_summary_block(raw_summary):
            if not isinstance(raw_summary, dict):
                raw_summary = {}
            return {str(k): _clip(v) for k, v in raw_summary.items()}

        if is_user_twin_state(user_signal):
            twin_user = extract_user_domain(user_signal)
            collaborative_context = extract_collaborative_context(user_signal)
            raw_user_state = {
                "intent": twin_user.get("intent", {}) if isinstance(twin_user.get("intent", {}), dict) else {},
                "ownership": twin_user.get("ownership", {}) if isinstance(twin_user.get("ownership", {}), dict) else {},
            }
            raw_collaborative_channels = collaborative_context.get("channels", {}) if isinstance(collaborative_context.get("channels", {}), dict) else {}
            cycle_block = twin_user.get("cycle_delta", {}) if isinstance(twin_user.get("cycle_delta", {}), dict) else {}
            intent_raw = raw_user_state.get("intent", {}).get("raw", {}) if isinstance(raw_user_state.get("intent", {}).get("raw", {}), dict) else {}
        else:
            raw_branches = user_signal.get("branches", {})
            if not isinstance(raw_branches, dict):
                raw_branches = {}
            raw_user_state = raw_branches.get("user_state", {}) if isinstance(raw_branches.get("user_state", {}), dict) else {}
            raw_collaborative = raw_branches.get("collaborative", {}) if isinstance(raw_branches.get("collaborative", {}), dict) else {}
            raw_collaborative_channels = raw_collaborative.get("channels", {}) if isinstance(raw_collaborative.get("channels", {}), dict) else {}
            cycle_block = {
                "observed_cycle_days": user_signal.get("observed_cycle_days", 90.0),
                "observed_cycle_days_by_category": user_signal.get("observed_cycle_days_by_category", {}),
                "observed_cycle_days_by_subcategory": user_signal.get("observed_cycle_days_by_subcategory", {}),
                "cycle_delta_ratio_by_category": user_signal.get("cycle_delta_ratio_by_category", {}),
                "cycle_delta_ratio_by_subcategory": user_signal.get("cycle_delta_ratio_by_subcategory", {}),
                "user_faster_than_expected_by_category": user_signal.get("user_faster_than_expected_by_category", {}),
                "user_faster_than_expected_by_subcategory": user_signal.get("user_faster_than_expected_by_subcategory", {}),
                "user_slower_than_expected_by_category": user_signal.get("user_slower_than_expected_by_category", {}),
                "user_slower_than_expected_by_subcategory": user_signal.get("user_slower_than_expected_by_subcategory", {}),
                "days_since_last_purchase": user_signal.get("days_since_last_purchase", -1),
                "budgets": user_signal.get("budgets", {}),
                "purchase_recency_by_category": user_signal.get("purchase_recency_by_category", {}),
                "purchase_recency_by_subcategory": user_signal.get("purchase_recency_by_subcategory", {}),
            }
            intent_raw = {}

        intent_clusters = _normalize_branch_clusters(raw_user_state.get("intent", {}).get("clusters", {}))
        ownership_clusters = _normalize_branch_clusters(raw_user_state.get("ownership", {}).get("clusters", {}))
        age_clusters = _normalize_branch_clusters(raw_collaborative_channels.get("age", {}).get("clusters", {}))
        gender_clusters = _normalize_branch_clusters(raw_collaborative_channels.get("gender", {}).get("clusters", {}))
        location_clusters = _normalize_branch_clusters(raw_collaborative_channels.get("location", {}).get("clusters", {}))
        age_summary = _normalize_summary_block(raw_collaborative_channels.get("age", {}).get("summary", {}))
        gender_summary = _normalize_summary_block(raw_collaborative_channels.get("gender", {}).get("summary", {}))
        location_summary = _normalize_summary_block(raw_collaborative_channels.get("location", {}).get("summary", {}))

        branches = {
            "user_state": {
                "intent": {
                    "summary": _normalize_summary_block(raw_user_state.get("intent", {}).get("summary", {})),
                    "clusters": intent_clusters,
                },
                "ownership": {
                    "summary": _normalize_summary_block(raw_user_state.get("ownership", {}).get("summary", {})),
                    "clusters": ownership_clusters,
                },
            },
            "collaborative": {
                "channels": {
                    "age": {"summary": age_summary, "clusters": age_clusters},
                    "gender": {"summary": gender_summary, "clusters": gender_clusters},
                    "location": {"summary": location_summary, "clusters": location_clusters},
                },
            },
        }

        return {
            "observed_cycle_days": float(np.clip(_safe_float(cycle_block.get("observed_cycle_days", 90.0), 90.0), 14.0, 365.0)),
            "observed_cycle_days_by_category": cycle_block.get("observed_cycle_days_by_category", {}) if isinstance(cycle_block.get("observed_cycle_days_by_category", {}), dict) else {},
            "observed_cycle_days_by_subcategory": cycle_block.get("observed_cycle_days_by_subcategory", {}) if isinstance(cycle_block.get("observed_cycle_days_by_subcategory", {}), dict) else {},
            "cycle_delta_ratio_by_category": cycle_block.get("cycle_delta_ratio_by_category", {}) if isinstance(cycle_block.get("cycle_delta_ratio_by_category", {}), dict) else {},
            "cycle_delta_ratio_by_subcategory": cycle_block.get("cycle_delta_ratio_by_subcategory", {}) if isinstance(cycle_block.get("cycle_delta_ratio_by_subcategory", {}), dict) else {},
            "user_faster_than_expected_by_category": cycle_block.get("user_faster_than_expected_by_category", {}) if isinstance(cycle_block.get("user_faster_than_expected_by_category", {}), dict) else {},
            "user_faster_than_expected_by_subcategory": cycle_block.get("user_faster_than_expected_by_subcategory", {}) if isinstance(cycle_block.get("user_faster_than_expected_by_subcategory", {}), dict) else {},
            "user_slower_than_expected_by_category": cycle_block.get("user_slower_than_expected_by_category", {}) if isinstance(cycle_block.get("user_slower_than_expected_by_category", {}), dict) else {},
            "user_slower_than_expected_by_subcategory": cycle_block.get("user_slower_than_expected_by_subcategory", {}) if isinstance(cycle_block.get("user_slower_than_expected_by_subcategory", {}), dict) else {},
            "days_since_last_purchase": int(_safe_float(cycle_block.get("days_since_last_purchase", -1), -1)),
            "budgets": cycle_block.get("budgets", {}) if isinstance(cycle_block.get("budgets", {}), dict) else {},
            "recency_by_category": cycle_block.get("purchase_recency_by_category", {}),
            "recency_by_subcategory": cycle_block.get("purchase_recency_by_subcategory", {}),
            "branches": branches,
            "active_view_product_ids": [int(pid) for pid in intent_raw.get("active_view_product_ids", []) if _safe_float(pid, np.nan) == _safe_float(pid, np.nan)],
            "active_wishlist_product_ids": [int(pid) for pid in intent_raw.get("active_wishlist_product_ids", []) if _safe_float(pid, np.nan) == _safe_float(pid, np.nan)],
            "active_cart_product_ids": [int(pid) for pid in intent_raw.get("active_cart_product_ids", []) if _safe_float(pid, np.nan) == _safe_float(pid, np.nan)],
            "intent_cluster_map": self._cluster_key_map(intent_clusters),
            "ownership_cluster_map": self._cluster_key_map(ownership_clusters),
            "coll_cluster_maps": {
                "age": self._cluster_key_map(age_clusters),
                "gender": self._cluster_key_map(gender_clusters),
                "location": self._cluster_key_map(location_clusters),
            },
            "coll_channel_support_scores": {
                channel: _clip(
                    (
                        branches.get("collaborative", {})
                        .get("channels", {})
                        .get(channel, {})
                        .get("summary", {})
                    ).get("channel_score", 0.0)
                )
                for channel in COLLAB_BASE_CHANNELS
            },
            "source_feature_cache": {},
            "intent_cluster_tensor_cache": {},
            "intent_cluster_array_cache": {},
            "ownership_cluster_array_cache": {},
            "collaborative_cluster_array_cache": {},
            "best_collaborative_signal_cache": {},
            "cluster_match_cache": {},
            "global_attr_signal_cache": {},
            "global_budget_alignment_cache": {},
            "intent_summary_array": self._summary_array(
                branches.get("user_state", {}).get("intent", {}).get("summary", {}),
                USER_STATE_INTENT_SUMMARY_FEATURES,
            ),
            "ownership_summary_array": self._summary_array(
                branches.get("user_state", {}).get("ownership", {}).get("summary", {}),
                USER_STATE_OWNERSHIP_SUMMARY_FEATURES,
            ),
            "collaborative_channel_support_tensor": self._build_collaborative_channel_support_tensor(
                branches.get("collaborative", {}) if isinstance(branches.get("collaborative", {}), dict) else {}
            ),
            "collaborative_channel_support_array": self._build_collaborative_channel_support_array(
                branches.get("collaborative", {}) if isinstance(branches.get("collaborative", {}), dict) else {}
            ),
        }

    def _compute_pair_metrics(self, cid, pid, state, viewed):
        brand, category, subcategory, size, color, material, keyword = self.product_attrs.get(pid, self.empty_attrs)
        user_state = state.get("branches", {}).get("user_state", {})
        ownership_clusters = (
            user_state.get("ownership", {}).get("clusters", [])
            if isinstance(user_state, dict)
            else []
        )
        source_spec = self._source_spec_for_pid(pid, category, subcategory)
        source_cache_key = source_spec.get("key") or f"{category}::{subcategory}"
        source_feature_cache = state.get("source_feature_cache", {})
        cached_source = source_feature_cache.get(source_cache_key) if isinstance(source_feature_cache, dict) else None
        if cached_source is None:
            intent_cluster = self._select_cluster(state.get("intent_cluster_map", {}), category, subcategory, scope="source_local", pid=pid)
            ownership_cluster = self._select_cluster(state.get("ownership_cluster_map", {}), category, subcategory, scope="source_local", pid=pid)
            source_purchase_score = _clip(self._cluster_scalar_signal(ownership_cluster, "source_strength"))
            source_purchase_recency = self._source_purchase_recency(state, category, subcategory, pid=pid)
            expected_profile = self._expected_cycle_profile(category, subcategory)
            expected_cycle_days = _safe_float(expected_profile.get("expected_cycle_days", np.nan), np.nan)
            expected_cycle_norm = (
                _clip(float(np.log1p(max(expected_cycle_days, 1.0))) / 6.5)
                if np.isfinite(expected_cycle_days) and expected_cycle_days > 0.0
                else 0.0
            )
            raw_cycle_ratio = self._cycle_delta_value(state, "cycle_delta_ratio", category, subcategory, pid=pid, default=1.0)
            user_cycle_ratio = _clip(float(np.clip(raw_cycle_ratio, 0.25, 4.0)) / 4.0)
            user_faster_than_expected = _clip(
                self._cycle_delta_value(state, "user_faster_than_expected", category, subcategory, pid=pid, default=0.0)
            )
            user_slower_than_expected = _clip(
                self._cycle_delta_value(state, "user_slower_than_expected", category, subcategory, pid=pid, default=0.0)
            )
            cluster_suppression = _clip(
                source_purchase_score
                * source_purchase_recency
                * (0.65 - 0.25 * user_faster_than_expected + 0.20 * user_slower_than_expected)
            )
            base_family_recovery = _clip(
                (1.0 - source_purchase_recency)
                * (
                    0.55
                    + 0.20 * expected_cycle_norm
                    + 0.15 * user_faster_than_expected
                    + 0.10 * (1.0 - user_slower_than_expected)
                )
            )
            post_purchase_phase = self._post_purchase_phase(state)
            adjacent_engagement = _clip(self._cluster_scalar_signal(intent_cluster, "adjacent_engagement"))
            interaction_boost = _clip(
                max(
                    _clip(self._cluster_scalar_signal(ownership_cluster, "interaction_boost")),
                    0.50 * post_purchase_phase * adjacent_engagement,
                )
            )
            source_family_recovery = _clip(
                min(
                    max(
                        base_family_recovery,
                        _clip(self._cluster_scalar_signal(ownership_cluster, "recovery")),
                        _clip(1.0 - cluster_suppression),
                    ),
                    1.0 - 0.55 * source_purchase_score * post_purchase_phase,
                )
            )
            intent_raw = user_state.get("intent", {}).get("raw", {}) if isinstance(user_state.get("intent", {}), dict) else {}
            family_key = self.product_family_lookup.get(pid)
            family_direct_intent = 0.0
            subcategory_rollup_intent = 0.0
            category_rollup_intent = 0.0
            cart_complement_signal = 0.0
            if isinstance(intent_raw, dict):
                fam_map = intent_raw.get("intent_weights_by_family", {}) or {}
                sub_map = intent_raw.get("intent_weights_by_subcategory", {}) or {}
                cat_map = intent_raw.get("intent_weights_by_category", {}) or {}
                cart_complement_map = intent_raw.get("cart_complement_weights", {}) or {}
                if family_key is not None:
                    family_direct_intent = float(_safe_float(fam_map.get(str(family_key), 0.0), 0.0))
                if subcategory is not None:
                    subcategory_rollup_intent = float(_safe_float(sub_map.get(str(subcategory), 0.0), 0.0))
                if category is not None:
                    category_rollup_intent = float(_safe_float(cat_map.get(str(category), 0.0), 0.0))
                if isinstance(cart_complement_map, dict):
                    direct_match = float(
                        max(
                            _safe_float(cart_complement_map.get(pid, 0.0), 0.0),
                            _safe_float(cart_complement_map.get(str(pid), 0.0), 0.0),
                        )
                    )
                    family_match = 0.0
                    subcategory_match = 0.0
                    category_match = 0.0
                    for raw_related_pid, raw_weight in cart_complement_map.items():
                        weight = max(_safe_float(raw_weight, 0.0), 0.0)
                        if weight <= 0.0:
                            continue
                        try:
                            related_pid = int(raw_related_pid)
                        except Exception:
                            continue
                        related_attrs = self.product_attrs.get(int(related_pid), self.empty_attrs)
                        related_category = related_attrs[1]
                        related_subcategory = related_attrs[2]
                        related_family = self.product_family_lookup.get(int(related_pid))
                        if family_key is not None and related_family == family_key:
                            family_match += weight
                        if subcategory is not None and related_subcategory == subcategory:
                            subcategory_match += weight
                        if category is not None and related_category == category:
                            category_match += weight
                    cart_complement_signal = _clip(
                        max(
                            direct_match,
                            family_match,
                            0.55 * subcategory_match,
                            0.30 * category_match,
                        )
                    )

            if family_direct_intent > 1e-8:
                family_intent_val = _clip(
                    family_direct_intent
                    + 0.20 * subcategory_rollup_intent
                    + 0.10 * category_rollup_intent
                )
            else:
                family_intent_val = _clip(max(subcategory_rollup_intent, 0.85 * category_rollup_intent))

            cluster_source_strength = _clip(self._cluster_scalar_signal(intent_cluster, "source_strength"))
            computed_source_intent = max(
                cluster_source_strength,
                _clip(family_intent_val),
                0.82 * cart_complement_signal,
            )
            cached_source = {
                "intent_cluster": intent_cluster,
                "ownership_cluster": ownership_cluster,
                "purchase_scope_clusters": {"source_local": ownership_cluster},
                "purchase_attr_cache": {},
                "source_budget_alignment_cache": {},
                "source_purchase_score": source_purchase_score,
                "source_intent_score": computed_source_intent,
                "family_local_intent": _clip(family_intent_val),
                "family_direct_intent": _clip(family_direct_intent),
                "subcategory_rollup_intent": _clip(subcategory_rollup_intent),
                "category_rollup_intent": _clip(category_rollup_intent),
                "cart_complement_signal": _clip(cart_complement_signal),
                "intent_view_engagement": _clip(self._cluster_scalar_signal(intent_cluster, "view_engagement")),
                "intent_adjacent_engagement": adjacent_engagement,
                "cluster_suppression": cluster_suppression,
                "source_family_recovery": source_family_recovery,
                "source_purchase_recency": source_purchase_recency,
                "expected_cycle_norm": expected_cycle_norm,
                "user_cycle_ratio": user_cycle_ratio,
                "user_faster_than_expected": user_faster_than_expected,
                "user_slower_than_expected": user_slower_than_expected,
                "interaction_boost": interaction_boost,
                "post_purchase_phase": post_purchase_phase,
            }
            if isinstance(source_feature_cache, dict):
                source_feature_cache[source_cache_key] = cached_source
        intent_cluster = cached_source.get("intent_cluster")
        ownership_cluster = cached_source.get("ownership_cluster")
        attr_values = {
            "brand": brand,
            "keyword": keyword,
            "size": size,
            "color": color,
            "material": material,
        }

        source_purchase_score = _clip(cached_source.get("source_purchase_score", 0.0))
        source_intent_score = _clip(cached_source.get("source_intent_score", 0.0))
        adjacent_engagement = _clip(cached_source.get("intent_adjacent_engagement", 0.0))
        source_family_recovery = _clip(cached_source.get("source_family_recovery", 0.0))
        source_purchase_recency = _clip(cached_source.get("source_purchase_recency", 0.0))
        expected_cycle_norm = _clip(cached_source.get("expected_cycle_norm", 0.0))
        user_cycle_ratio = _clip(cached_source.get("user_cycle_ratio", 0.0))
        user_faster_than_expected = _clip(cached_source.get("user_faster_than_expected", 0.0))
        user_slower_than_expected = _clip(cached_source.get("user_slower_than_expected", 0.0))
        cluster_suppression = _clip(cached_source.get("cluster_suppression", 0.0))

        attr_match_scores = {attr: 0.0 for attr in SCORE_ATTR_KEYS}

        for attr in SCORE_ATTR_KEYS:
            policy = self.signal_policies.get(attr, {})
            purchase_scope = policy.get("purchase_scope", policy.get("scope", "source_local"))
            attr_value = attr_values[attr]
            if attr_value is None:
                continue
            attr_value_key = _cacheable_attr_value(attr_value)
            if purchase_scope == "global":
                p_val = self._cached_aggregate_cluster_attr_signal(state, ownership_clusters, attr, attr_value)
            else:
                purchase_attr_cache = cached_source.get("purchase_attr_cache", {})
                purchase_scope_clusters = cached_source.get("purchase_scope_clusters", {})
                cache_key = (str(purchase_scope), str(attr), attr_value_key)
                p_val = purchase_attr_cache.get(cache_key) if isinstance(purchase_attr_cache, dict) else None
                if p_val is None:
                    has_scoped_cluster = isinstance(purchase_scope_clusters, dict) and purchase_scope in purchase_scope_clusters
                    scoped_cluster = purchase_scope_clusters.get(purchase_scope) if isinstance(purchase_scope_clusters, dict) else None
                    if not has_scoped_cluster:
                        scoped_cluster = self._select_cluster(state.get("ownership_cluster_map", {}), category, subcategory, scope=purchase_scope, pid=pid)
                        if isinstance(purchase_scope_clusters, dict):
                            purchase_scope_clusters[purchase_scope] = scoped_cluster
                    p_val = self._cached_cluster_signal_match(state, scoped_cluster, attr, attr_value)
                    if isinstance(purchase_attr_cache, dict):
                        purchase_attr_cache[cache_key] = float(p_val)
            attr_match_scores[attr] = _clip(max(_safe_float(p_val, 0.0), 0.0))

        intent_score = _clip(
            0.55 * source_intent_score
            + 0.30 * _clip(cached_source.get("intent_view_engagement", 0.0))
            + 0.15 * _clip(cached_source.get("intent_adjacent_engagement", 0.0))
        )

        item_price = _safe_float(self.item_price_lookup.get(pid, 0.0), 0.0)
        global_budget_cache = state.get("global_budget_alignment_cache", {})
        global_budget_alignment = global_budget_cache.get(item_price) if isinstance(global_budget_cache, dict) else None
        if global_budget_alignment is None:
            global_budget_alignment = self._global_budget_alignment(state["budgets"], item_price)
            if isinstance(global_budget_cache, dict):
                global_budget_cache[item_price] = float(global_budget_alignment)

        source_budget_alignment = None
        source_budget_cache = cached_source.get("source_budget_alignment_cache", {})
        if isinstance(source_budget_cache, dict) and item_price in source_budget_cache:
            source_budget_alignment = source_budget_cache.get(item_price)
        if source_budget_alignment is None:
            if isinstance(ownership_cluster, dict):
                source_budget_alignment = self._cluster_budget_alignment(ownership_cluster, item_price)
                if source_budget_alignment <= 0.0:
                    source_budget_alignment = None
            if source_budget_alignment is None:
                source_budget_alignment = self._source_budget_alignment(state["budgets"], category, subcategory, item_price, pid=pid)
            if isinstance(source_budget_cache, dict):
                source_budget_cache[item_price] = None if source_budget_alignment is None else float(source_budget_alignment)
        if source_budget_alignment is None:
            source_budget_alignment = global_budget_alignment
        price_norm = _clip(self.item_price_norm_lookup.get(pid, 0.0))

        same_days = self.order_days_since.get((cid, pid))
        same_item_penalty = _clip(_exp_decay(same_days, 30.0 + 130.0 * price_norm))
        substitute_penalty = self._substitute_neighbor_decay(cid, pid, price_norm)

        post_purchase_phase = _clip(cached_source.get("post_purchase_phase", self._post_purchase_phase(state)))
        intent_momentum = 0.24 * adjacent_engagement + 0.26 * post_purchase_phase * adjacent_engagement
        if pid in viewed:
            intent_momentum += 0.45
        intent_momentum = _clip(intent_momentum)
        cart_complement_signal = _clip(cached_source.get("cart_complement_signal", 0.0))
        interaction_boost = _clip(
            max(
                cached_source.get("interaction_boost", 0.0),
                0.85 * cart_complement_signal,
            )
        )

        return {
            "intent_score": float(intent_score),
            "source_intent_score": float(source_intent_score),
            "family_local_intent": float(_clip(cached_source.get("family_local_intent", 0.0))),
            "family_direct_intent": float(_clip(cached_source.get("family_direct_intent", 0.0))),
            "subcategory_rollup_intent": float(_clip(cached_source.get("subcategory_rollup_intent", 0.0))),
            "category_rollup_intent": float(_clip(cached_source.get("category_rollup_intent", 0.0))),
            "cart_complement_signal": float(cart_complement_signal),
            "source_family_recovery": float(source_family_recovery),
            "source_purchase_recency": float(source_purchase_recency),
            "expected_cycle_norm": float(expected_cycle_norm),
            "user_cycle_ratio": float(user_cycle_ratio),
            "user_faster_than_expected": float(user_faster_than_expected),
            "user_slower_than_expected": float(user_slower_than_expected),
            "same_item_penalty": float(same_item_penalty),
            "substitute_penalty": float(substitute_penalty),
            "cluster_suppression": float(cluster_suppression),
            "brand_match": float(attr_match_scores.get("brand", 0.0)),
            "keyword_match": float(attr_match_scores.get("keyword", 0.0)),
            "size_match": float(attr_match_scores.get("size", 0.0)),
            "color_match": float(attr_match_scores.get("color", 0.0)),
            "material_match": float(attr_match_scores.get("material", 0.0)),
            "source_purchase_score": float(source_purchase_score),
            "source_budget_alignment": float(source_budget_alignment),
            "global_budget_alignment": float(global_budget_alignment),
            "price_norm": float(price_norm),
            "intent_momentum": float(intent_momentum),
            "interaction_boost": float(interaction_boost),
        }

    def _build_samples(self, num_workers, progress_callback=None):
        workers = max(1, int(num_workers))
        total_users = int(len(self.client_ids))
        processed_users = 0

        def _emit_progress(force: bool = False):
            if progress_callback is None:
                return
            if not force and processed_users % 25 != 0:
                return
            progress_callback(processed_users, total_users)

        if workers == 1:
            blocks = []
            for cid in tqdm(self.client_ids, desc="Building dataset samples", leave=False):
                res = _build_samples_for_user(self, cid)
                if res is not None:
                    blocks.append(res)
                processed_users += 1
                _emit_progress()
            _emit_progress(force=True)
            return _concat_sample_column_blocks(blocks)

        adaptive_chunk = len(self.client_ids) // max(workers * 96, 1)
        chunk_size = max(4, min(32, adaptive_chunk if adaptive_chunk > 0 else 4))
        client_chunks = list(_chunk_sequence(list(self.client_ids), chunk_size))
        blocks = []
        tqdm.write(
            f"Dataset build: launching {workers} worker(s) "
            f"for {len(client_chunks)} user chunk(s) (chunk_size={chunk_size})."
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            initializer=_init_dataset_build_worker,
            initargs=(self,),
        ) as exc:
            future_chunk_sizes = {
                exc.submit(_build_samples_chunk, chunk): int(len(chunk))
                for chunk in client_chunks
            }
            for res_future in tqdm(
                concurrent.futures.as_completed(future_chunk_sizes),
                total=len(future_chunk_sizes),
                desc="Building dataset samples",
                leave=False,
            ):
                res = res_future.result()
                if res is not None:
                    blocks.append(res)
                processed_users += future_chunk_sizes[res_future]
                _emit_progress()
        processed_users = total_users
        _emit_progress(force=True)
        return _concat_sample_column_blocks(blocks)

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        (
            cid,
            pid,
            reg,
            rank,
            cached_user_idx,
        ) = self._sample_payload(idx)

        runtime_cache = getattr(self, "_runtime_materialization_cache", None)
        if not isinstance(runtime_cache, dict):
            runtime_cache = {}
            self._runtime_materialization_cache = runtime_cache
        if runtime_cache.get("cid") != cid:
            current_state = runtime_cache.get("state")
            if current_state is not None:
                self._clear_transient_user_runtime_caches(current_state)
            user_signal = self.get_user_twin_state(cid)
            state = self._get_user_pair_state(cid, user_signal=user_signal)
            runtime_cache["cid"] = cid
            runtime_cache["user_signal"] = user_signal
            runtime_cache["state"] = state
        user_signal = runtime_cache.get("user_signal")
        state = runtime_cache.get("state")
        metrics = self._sample_feature_metrics(idx)
        user_state, collaborative, pair_context = self.get_feature_bundle_arrays(
            cid,
            pid,
            user_signal=user_signal,
            state=state,
            metrics=metrics,
        )
        return {
            "user_idx": np.int32(cached_user_idx),
            "user_state": user_state,
            "collaborative": collaborative,
            "pair_context": {
                "context": pair_context,
            },
            "label_reg": np.float32(reg),
            "label_rank": np.float32(rank),
        }
