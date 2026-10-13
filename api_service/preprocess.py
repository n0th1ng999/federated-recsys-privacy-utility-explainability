import concurrent.futures
import os
import copy
import re
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from support.feature_contract import COLLAB_BASE_CHANNELS
from support.preprocess_contract import (
    BRAND_BAD_STREAK_SUPPRESS_THRESHOLD,
    BRAND_TRUST_DECAY,
    BRAND_TRUST_MAX_ABS,
    BUDGET_ATTR_SPECS,
    COMPLEMENT_ADJACENT_WEIGHT,
    COMPLEMENT_CART_ADJACENT_WEIGHT,
    COMPLEMENT_BOOST_MAX_AGE_DAYS,
    COMPLEMENT_BOOST_TAU_DAYS,
    INTENT_SOURCE_SPECS,
    INTENT_SOURCE_WEIGHT_MAP,
)
from support.signal_utils import (
    add_weighted_mean as _add_weighted_mean,
    age_bucket as _age_bucket,
    age_bucket_membership as _age_bucket_membership,
    cluster_key as _cluster_key,
    decay_signal as _decay_signal,
    derive_collaborative_channel_support as _derive_collaborative_channel_support,
    ensure_columns as _ensure_columns,
    family_neighbor_search as _family_neighbor_search,
    family_text_matrix as _family_text_matrix,
    is_real_source_token as _is_real_source_token,
    l2_normalize_rows as _l2_normalize_rows,
    norm_log_series as _norm_log_series,
    normalize_group_token as _normalize_group_token,
    normalize_order_status as _normalize_order_status,
    normalize_scalar_token as _normalize_scalar_token,
    normalize_split_tokens as _normalize_split_tokens,
    rating_to_sentiment as _rating_to_sentiment,
    safe_float as _to_float,
    safe_norm_entropy as _safe_norm_entropy,
    taxonomy_leaf_spec as _taxonomy_leaf_spec,
    to_list_value as _to_list_value,
)
from support.twin_state import (
    build_user_event,
    is_user_twin_state,
    make_empty_user_twin_state,
    set_catalog_context_ref,
    set_collaborative_context,
    to_utc_timestamp,
    utc_now_iso,
)


def _chunk_sequence(values, chunk_size):
    size = max(1, int(chunk_size))
    for start in range(0, len(values), size):
        yield values[start:start + size]

RUNTIME_ANALYTICS_DEFAULTS = {
    "TotalViews": 0.0,
    "UniqueViewers": 0.0,
    "TotalOrders": 0.0,
    "DeliveredOrders": 0.0,
    "ReturnedOrders": 0.0,
    "CancelledOrders": 0.0,
    "TotalUnits": 0.0,
    "AvgQuantityPerOrder": 0.0,
    "GrossRevenue": 0.0,
    "UniqueBuyers": 0.0,
    "AvgRating": 0.0,
    "RatingCount": 0.0,
    "PositiveRatingRate": 0.0,
    "ReviewCount": 0.0,
    "DaysSinceLastOrder": -1.0,
    "EffectivePrice": 0.0,
    "CurrentDiscount": 0.0,
    "AvgDiscount": 0.0,
    "DiscountStd": 0.0,
    "PromoActiveRate": 0.0,
}

_FAMILY_NOISE_PATTERN = re.compile(r"\b(STOCK\s*OFF|STOCKOFF|OUTLET|PROMO(CAO)?|PROMO)\b", re.IGNORECASE)
_FAMILY_SPEC_PATTERN = re.compile(
    r"\b(EN\s*ISO\s*\d+[A-Z0-9:\-\/]*|EN\s*\d+[A-Z0-9:\-\/]*|ISO\s*\d+[A-Z0-9:\-\/]*|S1P|S1|S2|S3|S4|S5|O1|O2|O3|O4|O5|SB|SRA|SRB|SRC|SR|WRU|WR|HRO|FO|HI|CI|ESD|CR|AN|CE)\b",
    re.IGNORECASE,
)
_FAMILY_STOP_TOKENS = {
    "DE", "DA", "DO", "DOS", "DAS", "COM", "SEM", "PARA", "POR", "EM", "NO", "NA", "NOS", "NAS",
    "E", "OU", "UN", "UND", "UNID", "PACK", "PCS", "PAR", "KIT", "MODELO", "REF", "REFERENCIA",
    "LOW", "MID", "HIGH", "ALTO", "ALTA", "BAIXO", "BAIXA", "CURTO", "CURTA", "LONGO", "LONGA",
}
_FAMILY_MEASURE_PATTERN = re.compile(r"^\d+(MM|CM|M|MT|MTS|ML|L|LT|LTS|GR|GRS|KG|UN)?$", re.IGNORECASE)


def _family_text_tokens(value):
    token = _normalize_scalar_token(value)
    if token is None:
        return []
    text = str(token).upper()
    text = _FAMILY_NOISE_PATTERN.sub(" ", text)
    text = _FAMILY_SPEC_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [part for part in text.split(" ") if part]


def _row_family_modifier_tokens(row):
    tokens = set()
    for value in (
        row.get("Brand"),
        row.get("Colors", row.get("Color")),
        row.get("Sizes", row.get("Size")),
        row.get("Materials", row.get("Material")),
        row.get("Certifications"),
    ):
        if value is None:
            continue
        values = _to_list_value(value) if not isinstance(value, (list, tuple, set)) else list(value)
        for raw_value in values:
            tokens.update(_family_text_tokens(raw_value))
    return tokens


def _family_core_tokens(value, modifier_tokens):
    out = []
    seen = set()
    for token in _family_text_tokens(value):
        if token in modifier_tokens or token in _FAMILY_STOP_TOKENS:
            continue
        if len(token) <= 1 or token.isdigit() or _FAMILY_MEASURE_PATTERN.match(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _family_label_from_row(row, leaf_spec):
    modifier_tokens = _row_family_modifier_tokens(row)
    name_tokens = _family_core_tokens(row.get("Name"), modifier_tokens)
    keyword_tokens = _family_core_tokens(row.get("Keywords"), modifier_tokens)
    tokens = name_tokens[:3]
    if len(tokens) < 3:
        for token in keyword_tokens:
            if token in tokens:
                continue
            tokens.append(token)
            if len(tokens) >= 3:
                break
    if tokens:
        return " ".join(tokens).lower()
    if _is_real_source_token(leaf_spec.get("subcategory")):
        return str(leaf_spec.get("subcategory")).strip().lower()
    if _is_real_source_token(leaf_spec.get("category")):
        return str(leaf_spec.get("category")).strip().lower()
    return None


def _build_family_artifacts(products_signal, complements=None, bought_together=None):
    if products_signal is None or products_signal.empty or "ProductID" not in products_signal.columns:
        return {}, {}, {}, {}

    work = products_signal.copy().reset_index(drop=True)
    work["ProductID"] = pd.to_numeric(work["ProductID"], errors="coerce")
    work = work.dropna(subset=["ProductID"]).copy()
    if work.empty:
        return {}, {}, {}, {}
    work["ProductID"] = work["ProductID"].astype(int)

    family_lookup = {}
    family_metadata = {}
    taxonomy_leaf_lookup = {}

    grouped_rows = {}
    for row in work.itertuples(index=False):
        row_dict = row._asdict()
        pid = int(row_dict["ProductID"])
        leaf_spec = _taxonomy_leaf_spec(row_dict.get("Category"), row_dict.get("Subcategory"))
        leaf_key = leaf_spec.get("key")
        if leaf_key is not None:
            taxonomy_leaf_lookup[pid] = str(leaf_key)
        label = _family_label_from_row(pd.Series(row_dict), leaf_spec)
        seed_key = f"{leaf_key or 'unknown'}::{label or pid}"
        grouped_rows.setdefault(seed_key, []).append(row_dict)

    for seed_key, rows in grouped_rows.items():
        rows = sorted(rows, key=lambda item: int(item["ProductID"]))
        anchor_pid = int(rows[0]["ProductID"])
        leaf_spec = _taxonomy_leaf_spec(rows[0].get("Category"), rows[0].get("Subcategory"))
        family_id = f"fam::{seed_key}"
        label = _family_label_from_row(pd.Series(rows[0]), leaf_spec) or str(anchor_pid)
        product_ids = [int(item["ProductID"]) for item in rows]
        for pid in product_ids:
            family_lookup[pid] = family_id
        family_metadata[family_id] = {
            "family_id": family_id,
            "family_anchor_product_id": anchor_pid,
            "product_ids": product_ids,
            "family_label": label,
            "taxonomy_leaf_key": leaf_spec.get("key"),
            "category": leaf_spec.get("category"),
            "subcategory": leaf_spec.get("subcategory"),
        }

    relation_scores = {}

    def _bump_relation(src_family, dst_family, relation_type, score, signals):
        if src_family is None or dst_family is None or src_family == dst_family:
            return
        relation_scores.setdefault(str(src_family), {})
        current = relation_scores[str(src_family)].get(str(dst_family))
        next_score = float(np.clip(_to_float(score, 0.0), 0.0, 1.0))
        entry = {
            "family_id": str(dst_family),
            "relation_type": str(relation_type),
            "score": next_score,
            "signals": list(signals),
            "taxonomy_leaf_key": family_metadata.get(str(dst_family), {}).get("taxonomy_leaf_key"),
        }
        if current is None or next_score > _to_float(current.get("score", 0.0), 0.0):
            relation_scores[str(src_family)][str(dst_family)] = entry

    complements = complements if isinstance(complements, dict) else {}
    bought_together = bought_together if isinstance(bought_together, dict) else {}
    for raw_pid, rels in complements.items():
        try:
            src_pid = int(raw_pid)
        except Exception:
            continue
        src_family = family_lookup.get(src_pid)
        for rid, strength in rels or []:
            try:
                dst_pid = int(rid)
            except Exception:
                continue
            _bump_relation(src_family, family_lookup.get(dst_pid), "complementary", 0.75 * _to_float(strength, 0.0), ["catalog_relation"])
    for raw_pid, rels in bought_together.items():
        try:
            src_pid = int(raw_pid)
        except Exception:
            continue
        src_family = family_lookup.get(src_pid)
        for rid, score in (rels or {}).items():
            try:
                dst_pid = int(rid)
            except Exception:
                continue
            _bump_relation(src_family, family_lookup.get(dst_pid), "complementary", 0.85 * _to_float(score, 0.0), ["bought_together"])

    family_ids = sorted(family_metadata.keys())
    if family_ids:
        family_text = pd.Series(
            [
                " ".join(
                    filter(
                        None,
                        [
                            str(family_metadata[fid].get("family_label") or "").strip(),
                            str(family_metadata[fid].get("subcategory") or "").strip(),
                            str(family_metadata[fid].get("category") or "").strip(),
                        ],
                    )
                )
                for fid in family_ids
            ]
        )
        vectors = _l2_normalize_rows(_family_text_matrix(family_text, max_features=64))
        sims, idxs = _family_neighbor_search(vectors, top_k=min(6, max(len(family_ids), 2)))
        for row_idx, src_family in enumerate(family_ids):
            src_leaf = family_metadata.get(src_family, {}).get("taxonomy_leaf_key")
            for score, neighbor_idx in zip(sims[row_idx].tolist(), idxs[row_idx].tolist()):
                if int(neighbor_idx) == row_idx:
                    continue
                dst_family = family_ids[int(neighbor_idx)]
                dst_leaf = family_metadata.get(dst_family, {}).get("taxonomy_leaf_key")
                sim = float(_to_float(score, 0.0))
                if src_leaf and dst_leaf and src_leaf == dst_leaf and sim >= 0.18:
                    _bump_relation(src_family, dst_family, "similar", 0.45 + 0.55 * sim, ["taxonomy_leaf", "text_similarity"])
                elif sim >= 0.35:
                    _bump_relation(src_family, dst_family, "similar", sim, ["text_similarity"])

    family_relations = {
        str(fid): sorted(
            rels.values(),
            key=lambda entry: float(_to_float(entry.get("score", 0.0), 0.0)),
            reverse=True,
        )
        for fid, rels in relation_scores.items()
    }

    return family_lookup, family_metadata, taxonomy_leaf_lookup, family_relations


def _build_contextual_brand_affinity(weight_map, brand_lookup, family_lookup, leaf_lookup, main_lookup):
    family_counts = {}
    parent_counts = {}
    main_counts = {}

    for pid, weight in weight_map.items():
        w = _to_float(weight, 0.0)
        if w <= 0.0:
            continue
        brand = brand_lookup.get(pid)
        if brand is None or pd.isna(brand):
            continue

        family_id = family_lookup.get(pid)
        if family_id is not None:
            fc = family_counts.setdefault(str(family_id), {})
            fc[brand] = fc.get(brand, 0.0) + w

        leaf_key = leaf_lookup.get(pid)
        if leaf_key is not None and not pd.isna(leaf_key):
            pc = parent_counts.setdefault(str(leaf_key), {})
            pc[brand] = pc.get(brand, 0.0) + w

        main_cat = main_lookup.get(pid)
        if main_cat is not None and not pd.isna(main_cat):
            mc = main_counts.setdefault(main_cat, {})
            mc[brand] = mc.get(brand, 0.0) + w

    def _normalize_nested(counts_map):
        out = {}
        for key, counts in counts_map.items():
            total = sum(counts.values())
            if total <= 0.0:
                continue
            out[key] = {k: float(v / total) for k, v in counts.items() if _to_float(v, 0.0) > 0.0}
        return out

    return {
        "by_family": _normalize_nested(family_counts),
        "by_parent": _normalize_nested(parent_counts),
        "by_main_category": _normalize_nested(main_counts),
    }


def _build_contextual_token_affinity(weight_map, value_lookup, family_lookup, leaf_lookup, main_lookup, split=False):
    family_counts = {}
    parent_counts = {}
    main_counts = {}

    for pid, weight in weight_map.items():
        w = _to_float(weight, 0.0)
        if w <= 0.0:
            continue
        raw_value = value_lookup.get(pid)
        if raw_value is None or pd.isna(raw_value):
            continue

        values = _to_list_value(raw_value) if split else [_normalize_scalar_token(raw_value)]
        values = [v for v in values if v]
        if not values:
            continue

        family_id = family_lookup.get(pid)
        leaf_key = leaf_lookup.get(pid)
        main_cat = main_lookup.get(pid)
        for value in values:
            if family_id is not None:
                fc = family_counts.setdefault(str(family_id), {})
                fc[value] = fc.get(value, 0.0) + w

            if leaf_key is not None and not pd.isna(leaf_key):
                pc = parent_counts.setdefault(str(leaf_key), {})
                pc[value] = pc.get(value, 0.0) + w

            if main_cat is not None and not pd.isna(main_cat):
                mc = main_counts.setdefault(main_cat, {})
                mc[value] = mc.get(value, 0.0) + w

    def _normalize_nested(counts_map):
        out = {}
        for key, counts in counts_map.items():
            total = sum(counts.values())
            if total <= 0.0:
                continue
            out[key] = {k: float(v / total) for k, v in counts.items() if _to_float(v, 0.0) > 0.0}
        return out

    return {
        "by_family": _normalize_nested(family_counts),
        "by_parent": _normalize_nested(parent_counts),
        "by_main_category": _normalize_nested(main_counts),
    }
def _load_optional_frame(provided_df, fallback_path):
    if provided_df is not None:
        return provided_df.copy()
    if fallback_path and os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)
    return pd.DataFrame()


def _safe_product_ids(values, product_id_set=None):
    allowed = set(product_id_set or [])
    out = []
    seen = set()
    for raw_value in values or []:
        try:
            pid = int(raw_value)
        except Exception:
            continue
        if allowed and pid not in allowed:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _increment_weight_map(target, key, delta):
    if key is None:
        return
    value = _to_float(delta, 0.0)
    if abs(value) <= 1e-12:
        return
    target[int(key)] = target.get(int(key), 0.0) + value


def _replace_active_products(target, key, product_ids):
    target[key] = [int(pid) for pid in product_ids]


def _median_gap_lookup(date_map):
    out = {}
    for key, dates in date_map.items():
        if len(dates) < 2:
            continue
        ordered = pd.Series(pd.to_datetime(dates, errors="coerce", utc=True)).dropna().sort_values(kind="mergesort")
        if len(ordered) < 2:
            continue
        gaps = ordered.diff().dt.days.dropna()
        gaps = gaps[gaps > 0]
        if gaps.empty:
            continue
        out[key] = float(np.median(gaps))
    return out


def _repeat_confidence_from_gap_count(gap_count):
    return float(np.clip(1.0 - np.exp(-max(int(gap_count), 0) / 4.0), 0.0, 1.0))


def _build_replenishment_profiles(orders_merged, products_signal):
    empty = {"category": {}, "subcategory": {}}
    if orders_merged.empty or "ProductID" not in orders_merged.columns or "OrderDate" not in orders_merged.columns:
        return empty

    work = orders_merged[["ClientID", "ProductID", "OrderDate"]].copy()
    if "Category" not in work.columns or "Subcategory" not in work.columns:
        product_cols = [c for c in ["ProductID", "Category", "Subcategory"] if c in products_signal.columns]
        if "ProductID" in product_cols:
            product_lookup = products_signal[product_cols].reset_index(drop=True).drop_duplicates("ProductID")
            work = work.merge(product_lookup, on="ProductID", how="left")
    work["OrderDate"] = pd.to_datetime(work["OrderDate"], errors="coerce", utc=True)
    work = work.dropna(subset=["OrderDate"])
    if work.empty:
        return empty

    profiles = {"category": {}, "subcategory": {}}
    for level, column in (("category", "Category"), ("subcategory", "Subcategory")):
        if column not in work.columns:
            continue
        family_gaps = {}
        for (_, family_value), group in work.groupby(["ClientID", column], dropna=True):
            family_token = _normalize_scalar_token(family_value)
            if not _is_real_source_token(family_token):
                continue
            ordered = group["OrderDate"].sort_values(kind="mergesort").drop_duplicates()
            if len(ordered) < 2:
                continue
            gaps = ordered.diff().dt.days.dropna()
            gaps = gaps[gaps > 0]
            if gaps.empty:
                continue
            family_gaps.setdefault(family_token, []).extend(float(v) for v in gaps.tolist() if pd.notna(v) and float(v) > 0.0)
        for family_token, gap_values in family_gaps.items():
            if not gap_values:
                continue
            expected_cycle_days = float(np.clip(np.median(gap_values), 7.0, 540.0))
            profiles[level][family_token] = {
                "expected_cycle_days": expected_cycle_days,
                "repeat_confidence": _repeat_confidence_from_gap_count(len(gap_values)),
            }
    return profiles

def _build_collaborative_group_channels(clients, user_group_cols=None):
    user_group_cols = user_group_cols or ["Age", "Gender", "Location"]
    cols_set = set(clients.columns)
    cids = clients.index.tolist()

    has_age = "Age" in cols_set and "Age" in user_group_cols
    has_gender = "Gender" in cols_set and "Gender" in user_group_cols
    has_location = "Location" in cols_set and "Location" in user_group_cols

    age_map = {cid: (_age_bucket(clients.at[cid, "Age"]) if has_age else "age_unknown") for cid in cids}
    age_weight_map = {
        cid: (_age_bucket_membership(clients.at[cid, "Age"]) if has_age else {"age_unknown": 1.0})
        for cid in cids
    }
    gender_map = {
        cid: (_normalize_group_token(clients.at[cid, "Gender"], "gender") if has_gender else "gender_unknown")
        for cid in cids
    }
    location_map = {
        cid: (_normalize_group_token(clients.at[cid, "Location"], "location") if has_location else "location_unknown")
        for cid in cids
    }

    channel_user_groups = {
        "age": age_map,
        "gender": gender_map,
        "location": location_map,
    }

    channel_user_group_weights = {
        "age": {cid: age_weight_map[cid] for cid in cids},
        "gender": {cid: {gender_map[cid]: 1.0} for cid in cids},
        "location": {cid: {location_map[cid]: 1.0} for cid in cids},
    }

    channel_group_members = {}
    for channel, user_membership in channel_user_group_weights.items():
        groups = {}
        for cid, memberships in user_membership.items():
            if not isinstance(memberships, dict):
                continue
            for gid, w in memberships.items():
                wf = _to_float(w, 0.0)
                if wf <= 0.0:
                    continue
                groups.setdefault(gid, {})[cid] = groups.setdefault(gid, {}).get(cid, 0.0) + wf
        channel_group_members[channel] = groups

    return channel_user_groups, channel_group_members

def _build_item_group_affinity(orders_merged, client_to_group):
    if orders_merged.empty or "ProductID" not in orders_merged.columns or "ClientID" not in orders_merged.columns:
        return {}

    work = orders_merged[["ClientID", "ProductID"] + ([ "OrderStatus"] if "OrderStatus" in orders_merged.columns else [])].copy()
    if "OrderStatus" in work.columns:
        work["OrderStatus"] = work["OrderStatus"].fillna("delivered").astype(str).apply(_normalize_order_status)
        work = work[work["OrderStatus"] != "cancelled"]

    expanded_rows = []
    for row in work.itertuples(index=False):
        cid = getattr(row, "ClientID")
        pid = int(getattr(row, "ProductID"))
        membership = client_to_group.get(cid)
        if isinstance(membership, dict):
            for gid, w in membership.items():
                wf = _to_float(w, 0.0)
                if wf > 0.0:
                    expanded_rows.append((pid, gid, wf))
        elif membership is not None and not pd.isna(membership):
            expanded_rows.append((pid, membership, 1.0))
    if not expanded_rows:
        return {}
    work_grouped = pd.DataFrame(expanded_rows, columns=["ProductID", "group_id", "group_weight"])

    counts = (
        work_grouped.groupby(["ProductID", "group_id"], as_index=False)["group_weight"]
        .sum()
        .rename(columns={"group_weight": "cnt"})
    )
    counts["total_cnt"] = counts.groupby("ProductID")["cnt"].transform("sum")
    counts["share"] = counts["cnt"] / counts["total_cnt"].replace(0, np.nan)
    counts["support"] = 1.0 - np.exp(-counts["total_cnt"] / 4.0)
    counts["aff"] = counts["share"] * counts["support"]

    item_group_affinity = {}
    for pid, gdf in counts.groupby("ProductID"):
        item_group_affinity[int(pid)] = {
            g: float(a)
            for g, a in zip(gdf["group_id"], gdf["aff"])
            if pd.notna(a)
        }

    return item_group_affinity

def _build_group_collaborative_maps(item_group_affinity, group_ids):
    group_ids = list(group_ids)
    maps = {gid: {} for gid in group_ids}
    for pid, aff_by_group in item_group_affinity.items():
        for gid, score in aff_by_group.items():
            if gid in maps:
                if _to_float(score, 0.0) > 0.0:
                    maps[gid][int(pid)] = float(score)
    return maps

def _build_cluster_signal_blocks(
    weight_map,
    *,
    brand_lookup,
    main_lookup,
    sub_lookup,
    family_lookup,
    leaf_lookup,
    size_lookup,
    color_lookup,
    material_lookup,
    keyword_lookup,
    include_attrs=True,
):
    categories_with_subcategories = {}
    for pid, category in main_lookup.items():
        subcategory = sub_lookup.get(pid)
        category_token = _normalize_scalar_token(category)
        subcategory_token = _normalize_scalar_token(subcategory)
        if not _is_real_source_token(category_token) or not _is_real_source_token(subcategory_token):
            continue
        categories_with_subcategories.setdefault(category_token, set()).add(subcategory_token)

    cluster_counts = {}
    cluster_meta = {}
    attr_counts = {} if include_attrs else None

    def _source_spec(pid):
        category_token = _normalize_scalar_token(main_lookup.get(pid))
        subcategory_token = _normalize_scalar_token(sub_lookup.get(pid))
        has_category = _is_real_source_token(category_token)
        has_subcategory = _is_real_source_token(subcategory_token)
        family_id = family_lookup.get(pid)
        leaf_key = leaf_lookup.get(pid)
        if family_id is not None:
            return {
                "level": "family",
                "value": str(family_id),
                "parent": str(leaf_key) if leaf_key is not None and not pd.isna(leaf_key) else None,
                "category": category_token if has_category else None,
                "subcategory": subcategory_token if has_subcategory else None,
                "key": f"family::{family_id}",
            }
        if leaf_key is not None and not pd.isna(leaf_key):
            return {
                "level": "taxonomy_leaf",
                "value": str(leaf_key),
                "parent": category_token if has_category else None,
                "category": category_token if has_category else None,
                "subcategory": subcategory_token if has_subcategory else None,
                "key": str(leaf_key),
            }
        if has_category and has_subcategory and categories_with_subcategories.get(category_token):
            return {
                "level": "subcategory",
                "value": subcategory_token,
                "parent": category_token,
                "category": category_token,
                "subcategory": subcategory_token,
                "key": _cluster_key("subcategory", subcategory_token),
            }
        return {
            "level": "category",
            "value": category_token if has_category else None,
            "parent": None,
            "category": category_token if has_category else None,
            "subcategory": None,
            "key": _cluster_key("category", category_token) if has_category else None,
        }

    for pid, raw_weight in weight_map.items():
        weight = _to_float(raw_weight, 0.0)
        if weight <= 0.0:
            continue

        spec = _source_spec(pid)
        cluster_id = spec.get("key")
        if cluster_id is None:
            continue

        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0.0) + weight
        cluster_meta[cluster_id] = {
            "level": spec["level"],
            "value": spec["value"],
            "parent": spec["parent"],
            "category": spec.get("category"),
            "subcategory": spec.get("subcategory"),
        }
        if include_attrs:
            cluster_attr = attr_counts.setdefault(cluster_id, {
                "brand": {},
                "size": {},
                "color": {},
                "material": {},
                "keyword": {},
            })

            brand = _normalize_scalar_token(brand_lookup.get(pid))
            if brand is not None:
                cluster_attr["brand"][brand] = cluster_attr["brand"].get(brand, 0.0) + weight

            for attr_name, lookup in (
                ("size", size_lookup),
                ("color", color_lookup),
                ("material", material_lookup),
                ("keyword", keyword_lookup),
            ):
                raw_value = lookup.get(pid)
                values = _to_list_value(raw_value) if raw_value is not None and not pd.isna(raw_value) else []
                for value in values:
                    token = _normalize_scalar_token(value)
                    if token is None:
                        continue
                    cluster_attr[attr_name][token] = cluster_attr[attr_name].get(token, 0.0) + weight

    out = {}
    for cluster_id, total in cluster_counts.items():
        if total <= 0.0:
            continue
        meta = cluster_meta.get(cluster_id, {})
        signals = {}
        if include_attrs:
            for attr_name, counts in attr_counts.get(cluster_id, {}).items():
                if not counts:
                    signals[attr_name] = {"scores": {}}
                    continue
                attr_total = float(sum(counts.values()))
                signals[attr_name] = {
                    "scores": {key: float(val / attr_total) for key, val in counts.items() if _to_float(val, 0.0) > 0.0},
                }

        out[cluster_id] = {
            "level": meta.get("level"),
            "value": meta.get("value"),
            "parent": meta.get("parent"),
            "category": meta.get("category"),
            "subcategory": meta.get("subcategory"),
            "key": cluster_id,
            "source": {
                "score": float(1.0 - np.exp(-total / 3.0)),
            },
            "signals": signals,
        }
    return out


def _cluster_source_score_map(raw_clusters):
    out = {}
    if not isinstance(raw_clusters, dict):
        return out
    for cluster_id, block in raw_clusters.items():
        if not isinstance(block, dict):
            continue
        source = block.get("source", {})
        if not isinstance(source, dict):
            source = {}
        out[str(cluster_id)] = float(np.clip(_to_float(source.get("score", 0.0), 0.0), 0.0, 1.0))
    return out


def _brand_affinity_with_trust(value, brand_trust, brand_conf):
    base = float(np.clip(_to_float(value, 0.0), 0.0, 1.0))
    trust = float(np.clip(_to_float(brand_trust, 0.0), -1.0, 1.0))
    conf = float(np.clip(_to_float(brand_conf, 0.0), 0.0, 1.0))
    trust_bias = trust * (0.35 + 0.65 * conf)
    bias_scale = 0.55 if trust_bias < 0.0 else 0.35
    return float(np.clip(base + bias_scale * trust_bias, -1.0, 1.0))


def _cluster_context_scores(context_root, level, value, parent, category=None, subcategory=None):
    if not isinstance(context_root, dict):
        return {}

    if level == "family":
        family_map = context_root.get("by_family", {})
        if isinstance(family_map, dict):
            scores = family_map.get(str(value), {})
            if isinstance(scores, dict) and scores:
                return {str(k): float(_to_float(v, 0.0)) for k, v in scores.items()}

    if level in {"family", "taxonomy_leaf"}:
        parent_map = context_root.get("by_parent", {})
        parent_key = parent if level == "family" else value
        if isinstance(parent_map, dict):
            scores = parent_map.get(parent_key, {})
            if isinstance(scores, dict) and scores:
                return {str(k): float(_to_float(v, 0.0)) for k, v in scores.items()}

    if level == "subcategory":
        sub_map = context_root.get("by_subcategory", {})
        if isinstance(sub_map, dict):
            scores = sub_map.get(value, {})
            if isinstance(scores, dict) and scores:
                return {str(k): float(_to_float(v, 0.0)) for k, v in scores.items()}

    fallback_category = category
    if not _is_real_source_token(fallback_category):
        if level == "subcategory":
            fallback_category = parent
        elif level == "category":
            fallback_category = value

    main_map = context_root.get("by_main_category", context_root.get("by_category", {}))
    if isinstance(main_map, dict) and _is_real_source_token(fallback_category):
        scores = main_map.get(fallback_category, {})
        if isinstance(scores, dict) and scores:
            return {str(k): float(_to_float(v, 0.0)) for k, v in scores.items()}

    return {}


def _export_branch_clusters(
    raw_clusters,
    *,
    branch_kind,
    brand_context,
    token_contexts,
    intent_source_components=None,
    brand_trust=None,
    brand_trust_confidence=None,
    budgets=None,
    recency_by_category=None,
    recency_by_subcategory=None,
):
    out = {}
    brand_trust = brand_trust if isinstance(brand_trust, dict) else {}
    brand_trust_confidence = brand_trust_confidence if isinstance(brand_trust_confidence, dict) else {}
    intent_source_components = intent_source_components if isinstance(intent_source_components, dict) else {}
    recency_by_category = recency_by_category if isinstance(recency_by_category, dict) else {}
    recency_by_subcategory = recency_by_subcategory if isinstance(recency_by_subcategory, dict) else {}

    for cluster_id, raw_block in (raw_clusters or {}).items():
        if not isinstance(raw_block, dict):
            continue
        level = raw_block.get("level")
        value = raw_block.get("value")
        parent = raw_block.get("parent")
        category = raw_block.get("category")
        subcategory = raw_block.get("subcategory")
        source = raw_block.get("source", {})
        if not isinstance(source, dict):
            source = {}

        signals = {}

        if branch_kind != "intent":
            brand_scores = _cluster_context_scores(
                brand_context,
                level,
                value,
                parent,
                category=category,
                subcategory=subcategory,
            )
            if brand_scores:
                signed_brand_scores = {}
                for brand, score in brand_scores.items():
                    signed_brand_scores[str(brand)] = _brand_affinity_with_trust(
                        score,
                        brand_trust.get(brand, 0.0),
                        brand_trust_confidence.get(brand, 0.0),
                    )
                signals["brand"] = {"scores": signed_brand_scores}

            for attr_name, context_root in token_contexts.items():
                attr_scores = _cluster_context_scores(
                    context_root,
                    level,
                    value,
                    parent,
                    category=category,
                    subcategory=subcategory,
                )
                if attr_scores:
                    signals[attr_name] = {"scores": attr_scores}

        signals["source_strength"] = float(np.clip(_to_float(source.get("score", 0.0), 0.0), 0.0, 1.0))

        if branch_kind == "intent":
            for signal_name, component_scores in intent_source_components.items():
                if not isinstance(component_scores, dict):
                    continue
                signals[signal_name] = float(np.clip(_to_float(component_scores.get(str(cluster_id), 0.0), 0.0), 0.0, 1.0))

        budget_target = _branch_cluster_budget_target(
            level,
            value,
            parent,
            budgets,
            category=category,
            subcategory=subcategory,
            fallback_global=False,
        )
        if branch_kind == "ownership" and np.isfinite(budget_target) and budget_target > 0.0:
            signals["budget"] = float(budget_target)

        out[str(cluster_id)] = {
            "key": str(cluster_id),
            "level": level,
            "value": value,
            "parent": parent,
            "category": category,
            "subcategory": subcategory,
            "source": {
                "score": float(np.clip(_to_float(source.get("score", 0.0), 0.0), 0.0, 1.0)),
            },
            "signals": signals,
        }

    return out


def _branch_cluster_budget_target(
    cluster_level,
    cluster_value,
    cluster_parent,
    budgets,
    *,
    category=None,
    subcategory=None,
    fallback_global=False,
):
    if not isinstance(budgets, dict):
        return None
    if cluster_level == "family":
        family_map = budgets.get("family", {})
        if isinstance(family_map, dict) and cluster_value in family_map:
            return _to_float(family_map.get(cluster_value), np.nan)
        leaf_map = budgets.get("taxonomy_leaf", {})
        if isinstance(leaf_map, dict) and cluster_parent in leaf_map:
            return _to_float(leaf_map.get(cluster_parent), np.nan)
    elif cluster_level == "taxonomy_leaf":
        leaf_map = budgets.get("taxonomy_leaf", {})
        if isinstance(leaf_map, dict) and cluster_value in leaf_map:
            return _to_float(leaf_map.get(cluster_value), np.nan)
    if cluster_level == "subcategory":
        sub_map = budgets.get("subcategory", {})
        if isinstance(sub_map, dict) and cluster_value in sub_map:
            return _to_float(sub_map.get(cluster_value), np.nan)
        cat_map = budgets.get("category", {})
        if isinstance(cat_map, dict) and cluster_parent in cat_map:
            return _to_float(cat_map.get(cluster_parent), np.nan)
    elif cluster_level == "category":
        cat_map = budgets.get("category", {})
        if isinstance(cat_map, dict) and cluster_value in cat_map:
            return _to_float(cat_map.get(cluster_value), np.nan)
    if _is_real_source_token(subcategory):
        sub_map = budgets.get("subcategory", {})
        if isinstance(sub_map, dict) and subcategory in sub_map:
            return _to_float(sub_map.get(subcategory), np.nan)
    if _is_real_source_token(category):
        cat_map = budgets.get("category", {})
        if isinstance(cat_map, dict) and category in cat_map:
            return _to_float(cat_map.get(category), np.nan)
    if fallback_global:
        return _to_float(budgets.get("global", np.nan), np.nan)
    return np.nan
def _finalize_branch_clusters(
    raw_clusters,
    *,
    branch_kind,
    budgets=None,
    recency_by_category=None,
    recency_by_subcategory=None,
):
    out = {}
    recency_by_category = recency_by_category if isinstance(recency_by_category, dict) else {}
    recency_by_subcategory = recency_by_subcategory if isinstance(recency_by_subcategory, dict) else {}

    for cluster_id, raw_block in (raw_clusters or {}).items():
        if not isinstance(raw_block, dict):
            continue
        level = raw_block.get("level")
        value = raw_block.get("value")
        parent = raw_block.get("parent")
        category = raw_block.get("category")
        subcategory = raw_block.get("subcategory")
        raw_signals = raw_block.get("signals", {})
        if not isinstance(raw_signals, dict):
            raw_signals = {}

        signals = {}
        for attr_name, attr_signal in raw_signals.items():
            if not isinstance(attr_signal, dict):
                signals[attr_name] = float(_to_float(attr_signal, 0.0))
                continue
            scores = attr_signal.get("scores", {})
            if not isinstance(scores, dict):
                scores = {}
            signals[attr_name] = {
                "scores": {str(k): float(_to_float(v, 0.0)) for k, v in scores.items()},
            }

        source = raw_block.get("source", {})
        if not isinstance(source, dict):
            source = {}
        if "source_strength" not in signals:
            signals["source_strength"] = float(np.clip(_to_float(source.get("score", 0.0), 0.0), 0.0, 1.0))

        if branch_kind == "ownership" and "budget" not in signals:
            budget_target = _branch_cluster_budget_target(
                level,
                value,
                parent,
                budgets,
                category=category,
                subcategory=subcategory,
                fallback_global=False,
            )
            if np.isfinite(budget_target) and budget_target > 0.0:
                signals["budget"] = float(budget_target)

        out[str(cluster_id)] = {
            "key": str(cluster_id),
            "level": level,
            "value": value,
            "parent": parent,
            "category": category,
            "subcategory": subcategory,
            "signals": signals,
        }
    return out


def _active_mass_summary(values, scale):
    count = len(values) if isinstance(values, (list, tuple, set)) else 0
    return float(np.clip(float(count) / max(float(scale), 1.0), 0.0, 1.0))


def _cluster_strength_values(clusters, *, top_k=25):
    cluster_values = clusters.values() if isinstance(clusters, dict) else clusters
    strengths = []
    for cluster in cluster_values or []:
        if not isinstance(cluster, dict):
            continue
        signals = cluster.get("signals", {})
        if not isinstance(signals, dict):
            continue
        strength = float(np.clip(_to_float(signals.get("source_strength", 0.0), 0.0), 0.0, 1.0))
        if strength > 0.0:
            strengths.append(strength)
    strengths.sort(reverse=True)
    return strengths[: max(1, int(top_k))]


def _cluster_diversity_summary(clusters, *, top_k=25):
    return float(np.clip(_safe_norm_entropy(_cluster_strength_values(clusters, top_k=top_k)), 0.0, 1.0))


def _cluster_focus_concentration_summary(clusters, *, top_k=25):
    strengths = _cluster_strength_values(clusters, top_k=top_k)
    if not strengths:
        return 0.0
    total = float(sum(strengths))
    if total <= 1e-12:
        return 0.0
    return float(np.clip(strengths[0] / total, 0.0, 1.0))


def _cluster_recency_coverage_summary(clusters):
    cluster_values = clusters.values() if isinstance(clusters, dict) else clusters
    rows = []
    for cluster in cluster_values or []:
        if not isinstance(cluster, dict):
            continue
        signals = cluster.get("signals", {})
        if not isinstance(signals, dict):
            continue
        rows.append(float(np.clip(_to_float(signals.get("recency", 0.0), 0.0), 0.0, 1.0)))
    if not rows:
        return 0.0
    active = sum(1.0 for value in rows if value > 1e-8)
    return float(np.clip(float(active) / max(float(len(rows)), 1.0), 0.0, 1.0))


def _post_purchase_phase_summary(days_since_last_purchase, purchase_interval_days):
    days = _to_float(days_since_last_purchase, np.nan)
    if not np.isfinite(days) or days < 0.0:
        return 0.0
    interval_days = float(np.clip(_to_float(purchase_interval_days, 90.0), 14.0, 365.0))
    tau = float(np.clip(0.55 * interval_days, 14.0, 120.0))
    return float(np.clip(_decay_signal(days, tau), 0.0, 1.0))


def _build_intent_summary(
    *,
    interest_strength,
    view_pct,
    adjacent_pct,
    intent_clusters,
    intent_raw,
):
    return {
        "interest_strength": float(np.clip(_to_float(interest_strength, 0.0) / 25.0, 0.0, 1.0)),
        "source_view_share": float(np.clip(_to_float(view_pct, 0.0), 0.0, 1.0)),
        "source_adjacent_share": float(np.clip(_to_float(adjacent_pct, 0.0), 0.0, 1.0)),
        "active_view_mass": _active_mass_summary(intent_raw.get("active_view_product_ids", []), 8.0),
        "active_wishlist_mass": _active_mass_summary(intent_raw.get("active_wishlist_product_ids", []), 6.0),
        "active_cart_mass": _active_mass_summary(intent_raw.get("active_cart_product_ids", []), 4.0),
        "source_diversity": _cluster_diversity_summary(intent_clusters),
        "focus_concentration": _cluster_focus_concentration_summary(intent_clusters),
    }


def _build_ownership_summary(
    *,
    ownership_strength,
    ownership_ratio,
    ownership_clusters,
    budgets,
):
    budgets = budgets if isinstance(budgets, dict) else {}
    budget_global = _to_float(budgets.get("global", np.nan), np.nan)
    category_budget_count = len(budgets.get("category", {})) if isinstance(budgets.get("category", {}), dict) else 0
    subcategory_budget_count = len(budgets.get("subcategory", {})) if isinstance(budgets.get("subcategory", {}), dict) else 0
    budget_slots = category_budget_count + subcategory_budget_count + (1 if np.isfinite(budget_global) and budget_global > 0.0 else 0)
    return {
        "ownership_strength": float(np.clip(_to_float(ownership_strength, 0.0) / 25.0, 0.0, 1.0)),
        "ownership_interest_ratio": float(np.clip(_to_float(ownership_ratio, 0.0) / 3.0, 0.0, 1.0)),
        "budget_global_norm": float(np.clip(np.log1p(budget_global) / 10.0, 0.0, 1.0)) if np.isfinite(budget_global) and budget_global > 0.0 else 0.0,
        "budget_coverage": float(np.clip(float(budget_slots) / 12.0, 0.0, 1.0)),
        "ownership_diversity": _cluster_diversity_summary(ownership_clusters),
        "recency_coverage": _cluster_recency_coverage_summary(ownership_clusters),
    }


def _expected_cycle_profile(replenishment_profiles, category, subcategory):
    profiles = replenishment_profiles if isinstance(replenishment_profiles, dict) else {}
    sub_profiles = profiles.get("subcategory", {}) if isinstance(profiles.get("subcategory", {}), dict) else {}
    cat_profiles = profiles.get("category", {}) if isinstance(profiles.get("category", {}), dict) else {}
    sub_token = _normalize_scalar_token(subcategory)
    cat_token = _normalize_scalar_token(category)
    if _is_real_source_token(sub_token) and isinstance(sub_profiles.get(sub_token), dict):
        return sub_profiles.get(sub_token)
    if _is_real_source_token(cat_token) and isinstance(cat_profiles.get(cat_token), dict):
        return cat_profiles.get(cat_token)
    return {}


def _cycle_delta_maps(observed_cycles, replenishment_profiles):
    observed_cycles = observed_cycles if isinstance(observed_cycles, dict) else {}
    ratio_map = {}
    faster_map = {}
    slower_map = {}
    for key, observed_days in observed_cycles.items():
        profile = replenishment_profiles.get(str(key), {}) if isinstance(replenishment_profiles, dict) else {}
        expected_days = _to_float((profile or {}).get("expected_cycle_days", np.nan), np.nan)
        observed_days = _to_float(observed_days, np.nan)
        if not np.isfinite(expected_days) or expected_days <= 0.0 or not np.isfinite(observed_days) or observed_days <= 0.0:
            continue
        ratio = float(np.clip(observed_days / max(expected_days, 1e-6), 0.25, 4.0))
        ratio_map[str(key)] = ratio
        faster_map[str(key)] = float(np.clip((1.0 - ratio) / 0.75, 0.0, 1.0))
        slower_map[str(key)] = float(np.clip((ratio - 1.0) / 1.5, 0.0, 1.0))
    return ratio_map, faster_map, slower_map


def _build_product_attr_tokens(products_signal):
    attr_tokens = {}
    for pid, row in products_signal.iterrows():
        brand = _normalize_scalar_token(row.get("Brand"))
        category = _normalize_scalar_token(row.get("Category"))
        subcategory = _normalize_scalar_token(row.get("Subcategory"))
        sizes = _normalize_split_tokens(row.get("Sizes", row.get("Size")))
        colors = _normalize_split_tokens(row.get("Colors", row.get("Color")))
        materials = _normalize_split_tokens(row.get("Materials", row.get("Material")))
        keywords = _normalize_split_tokens(row.get("Keywords", row.get("Keyword")))
        attr_tokens[int(pid)] = {
            "brand": brand,
            "category": category,
            "subcategory": subcategory,
            "size": sizes,
            "color": colors,
            "material": materials,
            "keyword": keywords,
        }
    return attr_tokens


def _build_bought_together_map(orders_merged, topk=40):
    if orders_merged.empty or "OrderID" not in orders_merged.columns or "ProductID" not in orders_merged.columns:
        return {}

    work = orders_merged[["OrderID", "ProductID"]].copy()
    work["ProductID"] = pd.to_numeric(work["ProductID"], errors="coerce")
    work = work.dropna(subset=["ProductID"])
    if work.empty:
        return {}

    pair_counts = {}
    for _, group in work.groupby("OrderID"):
        pids = sorted({int(pid) for pid in group["ProductID"].tolist()})
        if len(pids) < 2:
            continue
        for i, pid in enumerate(pids):
            for rid in pids[i + 1:]:
                pair_counts[(pid, rid)] = pair_counts.get((pid, rid), 0.0) + 1.0
                pair_counts[(rid, pid)] = pair_counts.get((rid, pid), 0.0) + 1.0

    by_item = {}
    for (pid, rid), count in pair_counts.items():
        rel = by_item.setdefault(pid, {})
        rel[rid] = rel.get(rid, 0.0) + float(count)

    out = {}
    for pid, rel in by_item.items():
        if not rel:
            continue
        vmax = max(rel.values())
        ranked = sorted(rel.items(), key=lambda kv: (-kv[1], kv[0]))[:max(1, int(topk))]
        out[int(pid)] = {int(rid): float(cnt / max(vmax, 1e-12)) for rid, cnt in ranked}
    return out


def _build_substitute_neighbor_map(substitutes):
    neighbor_map = {}
    for raw_pid, rels in (substitutes or {}).items():
        try:
            pid = int(raw_pid)
        except Exception:
            continue
        for raw_rid, raw_strength in rels:
            try:
                rid = int(raw_rid)
            except Exception:
                continue
            if pid == rid:
                continue
            strength = float(np.clip(_to_float(raw_strength, 0.0), 0.0, 1.0))
            if strength <= 0.0:
                continue
            neighbor_map.setdefault(pid, {})
            neighbor_map.setdefault(rid, {})
            neighbor_map[pid][rid] = max(neighbor_map[pid].get(rid, 0.0), strength)
            neighbor_map[rid][pid] = max(neighbor_map[rid].get(pid, 0.0), strength)
    return neighbor_map


def _build_runtime_product_analytics(
    products,
    orders_merged,
    views_exploded,
    price_hist_df,
):
    prod_analytics = pd.DataFrame(index=products.index)
    prod_analytics.index.name = "ProductID"

    if not views_exploded.empty:
        view_stats = views_exploded.groupby("ProductID").agg(
            TotalViews=("ClientID", "size"),
            UniqueViewers=("ClientID", "nunique"),
        )
        prod_analytics = prod_analytics.join(view_stats, how="left")

    if not orders_merged.empty and "ProductID" in orders_merged.columns:
        order_sig = orders_merged.copy()
        if "OrderStatus" in order_sig.columns:
            order_sig["OrderStatus"] = order_sig["OrderStatus"].fillna("delivered").astype(str).apply(_normalize_order_status)
        else:
            order_sig["OrderStatus"] = "delivered"
        if "Quantity" not in order_sig.columns:
            order_sig["Quantity"] = 1.0
        order_sig["Quantity"] = pd.to_numeric(order_sig["Quantity"], errors="coerce").fillna(1.0)
        if "UnitPrice" not in order_sig.columns:
            order_sig["UnitPrice"] = 0.0
        order_sig["UnitPrice"] = pd.to_numeric(order_sig["UnitPrice"], errors="coerce").fillna(0.0)
        order_sig["line_revenue"] = order_sig["Quantity"] * order_sig["UnitPrice"]

        prod_analytics["TotalOrders"] = order_sig.groupby("ProductID")["OrderID"].nunique()
        status_orders = (
            order_sig.groupby(["ProductID", "OrderStatus"])["OrderID"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(columns=["delivered", "returned", "cancelled"], fill_value=0.0)
        )
        non_cancelled = order_sig[order_sig["OrderStatus"] != "cancelled"]

        prod_analytics["DeliveredOrders"] = status_orders["delivered"]
        prod_analytics["ReturnedOrders"] = status_orders["returned"]
        prod_analytics["CancelledOrders"] = status_orders["cancelled"]
        prod_analytics["TotalUnits"] = non_cancelled.groupby("ProductID")["Quantity"].sum()
        prod_analytics["AvgQuantityPerOrder"] = (
            prod_analytics["TotalUnits"] / prod_analytics["TotalOrders"].replace(0, np.nan)
        )
        prod_analytics["GrossRevenue"] = non_cancelled.groupby("ProductID")["line_revenue"].sum()
        prod_analytics["UniqueBuyers"] = non_cancelled.groupby("ProductID")["ClientID"].nunique()

        if "Rating" in order_sig.columns:
            order_sig["Rating"] = pd.to_numeric(order_sig["Rating"], errors="coerce")
            rated = non_cancelled[non_cancelled["Rating"].notna()]
            prod_analytics["AvgRating"] = rated.groupby("ProductID")["Rating"].mean()
            prod_analytics["RatingCount"] = rated.groupby("ProductID")["Rating"].size()
            positive = rated.assign(_pos=(rated["Rating"] >= 4).astype(float)).groupby("ProductID")["_pos"].mean()
            prod_analytics["PositiveRatingRate"] = positive

        review_col = "ReviewText" if "ReviewText" in order_sig.columns else ("Review" if "Review" in order_sig.columns else None)
        if review_col is not None:
            review_mask = order_sig[review_col].fillna("").astype(str).str.strip() != ""
            review_count = order_sig[review_mask & (order_sig["OrderStatus"] != "cancelled")].groupby("ProductID").size()
            prod_analytics["ReviewCount"] = review_count

        if "OrderDate" in order_sig.columns:
            order_sig["OrderDate"] = pd.to_datetime(order_sig["OrderDate"], errors="coerce", utc=True)
            non_cancelled_dates = order_sig[order_sig["OrderStatus"] != "cancelled"]
            last_order = non_cancelled_dates.groupby("ProductID")["OrderDate"].max()
            days_since = (to_utc_timestamp() - last_order).dt.days
            prod_analytics["DaysSinceLastOrder"] = days_since

    if not price_hist_df.empty and {"ProductID", "Date"}.issubset(price_hist_df.columns):
        ph = price_hist_df.copy()
        ph["Date"] = pd.to_datetime(ph["Date"], errors="coerce", utc=True)
        latest_ph = ph.sort_values("Date").groupby("ProductID").tail(1)
        if "EffectivePrice" in latest_ph.columns:
            prod_analytics["EffectivePrice"] = latest_ph.set_index("ProductID")["EffectivePrice"]
        if "DiscountPercent" in latest_ph.columns:
            prod_analytics["CurrentDiscount"] = latest_ph.set_index("ProductID")["DiscountPercent"]
            prod_analytics["AvgDiscount"] = ph.groupby("ProductID")["DiscountPercent"].mean()
            prod_analytics["DiscountStd"] = ph.groupby("ProductID")["DiscountPercent"].std()
        if "PromoType" in ph.columns:
            promo_rate = ph.assign(_promo=(ph["PromoType"].fillna("none").astype(str).str.lower() != "none").astype(float))
            prod_analytics["PromoActiveRate"] = promo_rate.groupby("ProductID")["_promo"].mean()

    _ensure_columns(prod_analytics, RUNTIME_ANALYTICS_DEFAULTS)
    prod_analytics["DaysSinceLastOrder"] = prod_analytics["DaysSinceLastOrder"].fillna(-1.0)

    prod_analytics["ViewToOrderRate"] = (
        prod_analytics["DeliveredOrders"].fillna(0.0) / prod_analytics["TotalViews"].replace(0, np.nan)
    )
    delivered_or_returned = prod_analytics["DeliveredOrders"].fillna(0.0) + prod_analytics["ReturnedOrders"].fillna(0.0)
    prod_analytics["ReturnRate"] = prod_analytics["ReturnedOrders"].fillna(0.0) / delivered_or_returned.replace(0, np.nan)
    prod_analytics["CancelRate"] = (
        prod_analytics["CancelledOrders"].fillna(0.0) / prod_analytics["TotalOrders"].replace(0, np.nan)
    )

    views_norm = _norm_log_series(prod_analytics["TotalViews"].fillna(0.0))
    orders_norm = _norm_log_series(prod_analytics["DeliveredOrders"].fillna(0.0))
    rating_norm = np.clip((prod_analytics["AvgRating"].fillna(0.0).values - 1.0) / 4.0, 0.0, 1.0)
    quality_penalty = 1.0 - np.clip(prod_analytics["ReturnRate"].fillna(0.0).values, 0.0, 1.0)
    pop = 0.45 * views_norm + 0.55 * orders_norm
    pop = pop * (0.8 + 0.2 * rating_norm) * (0.8 + 0.2 * quality_penalty)
    prod_analytics["Popularity"] = np.clip(pop, 0.0, 1.0)

    return prod_analytics.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_catalog_context(
    *,
    products_signal,
    price_lookup,
    complements,
    substitutes,
    bought_together,
    substitute_neighbors,
    replenishment_profiles,
    product_family_lookup,
    family_metadata,
    family_relations,
    taxonomy_leaf_lookup,
):
    products = {}
    for raw_pid, row in products_signal.iterrows():
        pid = int(raw_pid)
        source = {
            "level": "subcategory" if _is_real_source_token(row.get("Subcategory")) else "category",
            "value": _normalize_scalar_token(row.get("Subcategory") if _is_real_source_token(row.get("Subcategory")) else row.get("Category")),
            "parent": _normalize_scalar_token(row.get("Category")) if _is_real_source_token(row.get("Subcategory")) else None,
        }
        products[str(pid)] = {
            "product_id": pid,
            "attrs": {
                "brand": _normalize_scalar_token(row.get("Brand")),
                "category": _normalize_scalar_token(row.get("Category")),
                "subcategory": _normalize_scalar_token(row.get("Subcategory")),
                "size": _normalize_split_tokens(row.get("Sizes", row.get("Size"))),
                "color": _normalize_split_tokens(row.get("Colors", row.get("Color"))),
                "material": _normalize_split_tokens(row.get("Materials", row.get("Material"))),
                "keyword": _normalize_split_tokens(row.get("Keywords", row.get("Keyword"))),
            },
            "price": float(_to_float(price_lookup.get(pid, 0.0), 0.0)),
            "popularity": float(_to_float(row.get("Popularity", 0.0), 0.0)),
            "visibility": {
                "post_status": _normalize_scalar_token(row.get("PostStatus")),
                "catalog_visibility": _normalize_scalar_token(row.get("CatalogVisibility")),
            },
            "stock": {
                "quantity": _to_float(row.get("Stock", row.get("StockQuantity", 0.0)), 0.0),
                "status": _normalize_scalar_token(row.get("StockStatus")),
                "is_available": row.get("IsAvailable"),
            },
            "source": source,
        }

    return {
        "schema_version": 1,
        "mode": "catalog_context",
        "updated_at": utc_now_iso(),
        "relations": {
            "complements": {str(int(pid)): [(int(rid), float(_to_float(strength, 0.0))) for rid, strength in rels] for pid, rels in (complements or {}).items()},
            "substitutes": {str(int(pid)): [(int(rid), float(_to_float(strength, 0.0))) for rid, strength in rels] for pid, rels in (substitutes or {}).items()},
            "substitute_neighbors": {str(int(pid)): {str(int(rid)): float(_to_float(strength, 0.0)) for rid, strength in rels.items()} for pid, rels in (substitute_neighbors or {}).items()},
            "bought_together": {str(int(pid)): {str(int(rid)): float(_to_float(score, 0.0)) for rid, score in rels.items()} for pid, rels in (bought_together or {}).items()},
        },
        "product_attrs": {
            str(int(pid)): {
                "brand": values.get("brand"),
                "category": values.get("category"),
                "subcategory": values.get("subcategory"),
                "size": list(values.get("size", ())),
                "color": list(values.get("color", ())),
                "material": list(values.get("material", ())),
                "keyword": list(values.get("keyword", ())),
            }
            for pid, values in _build_product_attr_tokens(products_signal).items()
        },
        "source_dynamics": {
            "products": products,
        },
        "family": {
            "product_family_lookup": {
                str(int(pid)): str(fid)
                for pid, fid in (product_family_lookup or {}).items()
                if fid is not None
            },
            "metadata": copy.deepcopy(family_metadata or {}),
            "relations": copy.deepcopy(family_relations or {}),
            "taxonomy_leaf_lookup": {
                str(int(pid)): str(leaf_key)
                for pid, leaf_key in (taxonomy_leaf_lookup or {}).items()
                if leaf_key is not None
            },
        },
        "replenishment_profiles": {
            "category": copy.deepcopy((replenishment_profiles or {}).get("category", {})),
            "subcategory": copy.deepcopy((replenishment_profiles or {}).get("subcategory", {})),
        },
        "meta": {
            "freshness": "snapshot",
            "confidence": 1.0,
            "provenance": "preprocess_data_generic",
        },
    }


def _build_history_user_events(
    *,
    cid,
    now_ts,
    user_items,
    user_views,
    complements,
    product_id_set,
):
    events = []
    valid_products = set(product_id_set or [])
    if user_views:
        for dt, pids in user_views.get("rows", []):
            safe_pids = _safe_product_ids(pids, valid_products)
            if not safe_pids:
                continue
            dt_utc = to_utc_timestamp(dt) if pd.notna(dt) else pd.NaT
            days = max(int((now_ts - dt_utc).days) if pd.notna(dt_utc) else 0, 0)
            rec = float(np.exp(-days / 120.0))
            events.append(
                build_user_event(
                    "view",
                    occurred_at=dt_utc if pd.notna(dt_utc) else now_ts,
                    origin="historical",
                    user_id=cid,
                    payload={"product_ids": safe_pids, "weight": rec},
                )
            )

    if user_items:
        for pid, status, qty_raw, rating, paid_unit_price_raw, odate in user_items:
            pid = int(pid)
            if valid_products and pid not in valid_products:
                continue
            odate_utc = to_utc_timestamp(odate) if pd.notna(odate) else pd.NaT
            days = max(int((now_ts - odate_utc).days) if pd.notna(odate_utc) else 0, 0)
            complement_decay = _decay_signal(days, COMPLEMENT_BOOST_TAU_DAYS) if days <= COMPLEMENT_BOOST_MAX_AGE_DAYS else 0.0
            events.append(
                build_user_event(
                    "purchase",
                    occurred_at=odate_utc if pd.notna(odate_utc) else now_ts,
                    origin="historical",
                    user_id=cid,
                    product_id=pid,
                    payload={
                        "status": status,
                        "quantity": _to_float(qty_raw, 1.0),
                        "rating": rating,
                        "paid_unit_price": _to_float(paid_unit_price_raw, np.nan),
                        "complement_relations": complements.get(pid, []),
                        "complement_decay": complement_decay,
                    },
                )
            )
    events.sort(key=lambda row: row.get("occurred_at") or "")
    return events


def _apply_user_event_to_twin_state(twin_state, catalog_context, event, transition_ctx):
    if not is_user_twin_state(twin_state):
        twin_state = make_empty_user_twin_state(
            user_id=event.get("user_id"),
            updated_at=event.get("occurred_at"),
            provenance="event_replay",
        )

    payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
    state_user = twin_state.setdefault("state", {}).setdefault("user", {})
    intent_state = state_user.setdefault("intent", {"summary": {}, "clusters": {}, "raw": {}})
    ownership_state = state_user.setdefault("ownership", {"summary": {}, "clusters": {}, "raw": {}})
    cycle_state = state_user.setdefault("cycle_delta", {"raw": {}})
    meta = twin_state.setdefault("state", {}).setdefault("meta", {})
    intent_raw = intent_state.setdefault("raw", {})
    ownership_raw = ownership_state.setdefault("raw", {})
    cycle_raw = cycle_state.setdefault("raw", {})

    for key in (
        "intent_weights",
        "view_weights",
        "adjacent_weights",
        "wishlist_weights",
        "cart_weights",
        "cart_complement_weights",
    ):
        intent_raw.setdefault(key, {})
    for key in (
        "active_view_product_ids",
        "active_wishlist_product_ids",
        "active_cart_product_ids",
    ):
        intent_raw.setdefault(key, [])
    for key in (
        "purchase_weights",
        "paid_price_by_pid",
        "brand_trust",
        "brand_bad_streak",
        "brand_interaction_count",
    ):
        ownership_raw.setdefault(key, {})
    for key in (
        "purchase_dates_by_category",
        "purchase_dates_by_subcategory",
        "purchase_dates_by_family",
        "purchase_dates_by_taxonomy_leaf",
        "order_dates",
    ):
        cycle_raw.setdefault(key, {} if "by_" in key else [])

    field_lookup = transition_ctx.get("field_lookup", {})
    price_lookup = transition_ctx.get("price_lookup", {})
    brand_lookup = field_lookup.get("Brand", {})
    category_lookup = field_lookup.get("Category", {})
    subcategory_lookup = field_lookup.get("Subcategory", {})
    family_lookup = field_lookup.get("FamilyID", {})
    taxonomy_leaf_lookup = field_lookup.get("TaxonomyLeafKey", {})
    valid_products = set(transition_ctx.get("product_id_set", []))
    kind = str(event.get("kind") or "").strip().lower()
    complement_lookup = ((catalog_context or {}).get("relations", {}) or {}).get("complements", {})

    def _trust_multiplier(pid):
        brand_val = brand_lookup.get(pid)
        if brand_val is None or pd.isna(brand_val):
            return 1.0
        trust = _to_float(ownership_raw["brand_trust"].get(brand_val, 0.0), 0.0)
        count = _to_float(ownership_raw["brand_interaction_count"].get(brand_val, 0.0), 0.0)
        conf = float(np.clip(1.0 - np.exp(-count / 3.0), 0.0, 1.0))
        multiplier = float(np.clip(1.0 + 0.35 * trust * (0.35 + 0.65 * conf), 0.2, 1.8))
        bad_streak = int(ownership_raw["brand_bad_streak"].get(brand_val, 0))
        if trust < 0.0 and bad_streak >= BRAND_BAD_STREAK_SUPPRESS_THRESHOLD:
            multiplier *= 0.6
        return float(np.clip(multiplier, 0.2, 1.8))

    def _update_brand_trust(pid, event_sentiment):
        brand_val = brand_lookup.get(pid)
        if brand_val is None or pd.isna(brand_val):
            return
        prev_trust = _to_float(ownership_raw["brand_trust"].get(brand_val, 0.0), 0.0)
        prev_bad = int(ownership_raw["brand_bad_streak"].get(brand_val, 0))
        updated = BRAND_TRUST_DECAY * prev_trust + (1.0 - BRAND_TRUST_DECAY) * event_sentiment
        ownership_raw["brand_trust"][brand_val] = float(np.clip(updated, -BRAND_TRUST_MAX_ABS, BRAND_TRUST_MAX_ABS))
        ownership_raw["brand_interaction_count"][brand_val] = int(ownership_raw["brand_interaction_count"].get(brand_val, 0)) + 1
        if event_sentiment < -0.2:
            ownership_raw["brand_bad_streak"][brand_val] = prev_bad + 1
        elif event_sentiment > 0.2:
            ownership_raw["brand_bad_streak"][brand_val] = max(prev_bad - 1, 0)
        else:
            ownership_raw["brand_bad_streak"][brand_val] = prev_bad

    def _rebuild_cart_complement_weights():
        rebuilt = {}
        for cart_pid in [int(value) for value in intent_raw.get("active_cart_product_ids", [])]:
            for rid, strength in complement_lookup.get(str(cart_pid), []):
                try:
                    related_pid = int(rid)
                except Exception:
                    continue
                if valid_products and related_pid not in valid_products:
                    continue
                _increment_weight_map(rebuilt, related_pid, COMPLEMENT_CART_ADJACENT_WEIGHT * _to_float(strength, 0.0))
        intent_raw["cart_complement_weights"] = {int(pid): float(weight) for pid, weight in rebuilt.items() if _to_float(weight, 0.0) > 0.0}

    occurred_at = to_utc_timestamp(event.get("occurred_at")) if event.get("occurred_at") is not None else pd.NaT
    if kind == "view":
        weight = _to_float(payload.get("weight", 1.0), 1.0)
        product_ids = _safe_product_ids(payload.get("product_ids", []), valid_products)
        _replace_active_products(intent_raw, "active_view_product_ids", product_ids)
        for pid in product_ids:
            weighted = weight * _trust_multiplier(pid)
            _increment_weight_map(intent_raw["view_weights"], pid, weight)
            _increment_weight_map(intent_raw["intent_weights"], pid, INTENT_SOURCE_WEIGHT_MAP["view"] * weighted)
    elif kind == "wishlist_add":
        weight = _to_float(payload.get("weight", 1.0), 1.0)
        product_ids = _safe_product_ids(payload.get("product_ids", [event.get("product_id")]), valid_products)
        _replace_active_products(intent_raw, "active_wishlist_product_ids", product_ids)
        for pid in product_ids:
            weighted = weight * _trust_multiplier(pid)
            _increment_weight_map(intent_raw["wishlist_weights"], pid, weight)
            _increment_weight_map(intent_raw["intent_weights"], pid, 0.55 * weighted)
    elif kind == "wishlist_remove":
        product_ids = _safe_product_ids(payload.get("product_ids", [event.get("product_id")]), valid_products)
        if product_ids:
            active = [int(value) for value in intent_raw.get("active_wishlist_product_ids", []) if int(value) not in set(product_ids)]
            intent_raw["active_wishlist_product_ids"] = active
            for pid in product_ids:
                current_wishlist = _to_float(intent_raw["wishlist_weights"].get(pid, 0.0), 0.0)
                current_intent = _to_float(intent_raw["intent_weights"].get(pid, 0.0), 0.0)
                if current_wishlist > 0.0:
                    intent_raw["wishlist_weights"][pid] = float(max(0.0, 0.35 * current_wishlist))
                if current_intent > 0.0:
                    intent_raw["intent_weights"][pid] = float(max(0.0, current_intent - 0.45 * current_wishlist))
    elif kind == "cart_add":
        weight = _to_float(payload.get("weight", 1.0), 1.0)
        product_ids = _safe_product_ids(payload.get("product_ids", [event.get("product_id")]), valid_products)
        _replace_active_products(intent_raw, "active_cart_product_ids", product_ids)
        for pid in product_ids:
            weighted = weight * _trust_multiplier(pid)
            _increment_weight_map(intent_raw["cart_weights"], pid, weight)
            _increment_weight_map(intent_raw["intent_weights"], pid, 0.85 * weighted)
            for rid, strength in complement_lookup.get(str(pid), []):
                _increment_weight_map(intent_raw["adjacent_weights"], int(rid), 0.70 * weight * _to_float(strength, 0.0))
                _increment_weight_map(intent_raw["intent_weights"], int(rid), INTENT_SOURCE_WEIGHT_MAP["adjacent"] * 0.70 * weight * _to_float(strength, 0.0))
        _rebuild_cart_complement_weights()
    elif kind == "cart_remove":
        product_ids = _safe_product_ids(payload.get("product_ids", [event.get("product_id")]), valid_products)
        if product_ids:
            active = [int(value) for value in intent_raw.get("active_cart_product_ids", []) if int(value) not in set(product_ids)]
            intent_raw["active_cart_product_ids"] = active
            for pid in product_ids:
                current_cart = _to_float(intent_raw["cart_weights"].get(pid, 0.0), 0.0)
                current_intent = _to_float(intent_raw["intent_weights"].get(pid, 0.0), 0.0)
                if current_cart > 0.0:
                    intent_raw["cart_weights"][pid] = float(max(0.0, 0.25 * current_cart))
                if current_intent > 0.0:
                    intent_raw["intent_weights"][pid] = float(max(0.0, current_intent - 0.60 * current_cart))
        _rebuild_cart_complement_weights()
    elif kind == "cart_update":
        weight = max(_to_float(payload.get("weight", 1.0), 1.0), 0.0)
        product_ids = _safe_product_ids(payload.get("product_ids", [event.get("product_id")]), valid_products)
        _replace_active_products(intent_raw, "active_cart_product_ids", product_ids)
        for pid in product_ids:
            current = _to_float(intent_raw["cart_weights"].get(pid, 0.0), 0.0)
            intent_raw["cart_weights"][pid] = float(max(current, weight))
            _increment_weight_map(intent_raw["intent_weights"], pid, 0.55 * weight * _trust_multiplier(pid))
        _rebuild_cart_complement_weights()
    elif kind == "purchase":
        pid = event.get("product_id")
        if pid is None:
            return twin_state
        pid = int(pid)
        status = payload.get("status", "delivered")
        qty = max(_to_float(payload.get("quantity", 1.0), 1.0), 0.0)
        rating = payload.get("rating", np.nan)
        status_factor = 1.0 if status == "delivered" else 0.45 if status == "returned" else 0.0 if status == "cancelled" else 0.8
        rating_factor = 1.0 if pd.isna(rating) else float(np.clip(0.6 + float(rating) / 5.0, 0.3, 1.6))
        event_sentiment = float(
            np.clip(
                0.65 * _rating_to_sentiment(rating)
                + 0.35 * (0.1 if status == "delivered" else -0.45 if status == "returned" else -0.35 if status == "cancelled" else 0.0),
                -1.0,
                1.0,
            )
        )
        purchase_weight = status_factor * rating_factor * (1.0 + 0.12 * np.log1p(max(qty, 0.0))) * _trust_multiplier(pid)
        if status_factor > 0.0:
            _increment_weight_map(ownership_raw["purchase_weights"], pid, purchase_weight)
            paid_unit_price = _to_float(payload.get("paid_unit_price", np.nan), np.nan)
            if np.isnan(paid_unit_price) or paid_unit_price <= 0.0:
                paid_unit_price = _to_float(price_lookup.get(pid, np.nan), np.nan)
            if np.isfinite(paid_unit_price) and paid_unit_price > 0.0:
                num, den = ownership_raw["paid_price_by_pid"].get(pid, (0.0, 0.0))
                ownership_raw["paid_price_by_pid"][pid] = [num + paid_unit_price * max(qty, 1.0), den + max(qty, 1.0)]

            if pd.notna(occurred_at):
                cycle_raw["order_dates"].append(occurred_at)
                cat_v = category_lookup.get(pid)
                sub_v = subcategory_lookup.get(pid)
                fam_v = family_lookup.get(pid)
                leaf_v = taxonomy_leaf_lookup.get(pid)
                if cat_v is not None and not pd.isna(cat_v):
                    cycle_raw["purchase_dates_by_category"].setdefault(cat_v, []).append(occurred_at)
                if sub_v is not None and not pd.isna(sub_v):
                    cycle_raw["purchase_dates_by_subcategory"].setdefault(sub_v, []).append(occurred_at)
                if fam_v is not None and not pd.isna(fam_v):
                    cycle_raw["purchase_dates_by_family"].setdefault(str(fam_v), []).append(occurred_at)
                if leaf_v is not None and not pd.isna(leaf_v):
                    cycle_raw["purchase_dates_by_taxonomy_leaf"].setdefault(str(leaf_v), []).append(occurred_at)

            for rid, strength in payload.get("complement_relations", []):
                boost = COMPLEMENT_ADJACENT_WEIGHT * _to_float(strength, 0.0) * _to_float(payload.get("complement_decay", 0.0), 0.0) * purchase_weight
                _increment_weight_map(intent_raw["adjacent_weights"], int(rid), boost)
                _increment_weight_map(intent_raw["intent_weights"], int(rid), INTENT_SOURCE_WEIGHT_MAP["adjacent"] * boost)

        _update_brand_trust(pid, event_sentiment)
        if int(pid) in intent_raw["active_wishlist_product_ids"]:
            intent_raw["active_wishlist_product_ids"] = [value for value in intent_raw["active_wishlist_product_ids"] if int(value) != int(pid)]
        if int(pid) in intent_raw["active_cart_product_ids"]:
            intent_raw["active_cart_product_ids"] = [value for value in intent_raw["active_cart_product_ids"] if int(value) != int(pid)]
    elif kind == "rating":
        pid = event.get("product_id")
        if pid is None:
            return twin_state
        _update_brand_trust(int(pid), _rating_to_sentiment(payload.get("rating", np.nan)))

    meta["updated_at"] = utc_now_iso(occurred_at if pd.notna(occurred_at) else None)
    meta["freshness"] = "historical" if event.get("origin") == "historical" else "live"
    meta["provenance"] = "event_replay"
    return twin_state


def _refresh_user_twin_state(
    twin_state,
    *,
    now_ts,
    transition_ctx,
    collaborative_group_ids=None,
    collaborative_maps_by_channel=None,
    materialize_collaborative_channels=True,
):
    if not is_user_twin_state(twin_state):
        return twin_state
    now_ts = to_utc_timestamp(now_ts)

    field_lookup = transition_ctx.get("field_lookup", {})
    price_lookup = transition_ctx.get("price_lookup", {})
    main_category_lookup = field_lookup.get("Category", {})
    subcategory_lookup = field_lookup.get("Subcategory", {})
    brand_lookup = field_lookup.get("Brand", {})
    family_lookup = field_lookup.get("FamilyID", {})
    leaf_lookup = field_lookup.get("TaxonomyLeafKey", {})
    replenishment_profiles = transition_ctx.get("replenishment_profiles", {})
    category_profiles = (
        replenishment_profiles.get("category", {})
        if isinstance(replenishment_profiles.get("category", {}), dict)
        else {}
    )
    subcategory_profiles = (
        replenishment_profiles.get("subcategory", {})
        if isinstance(replenishment_profiles.get("subcategory", {}), dict)
        else {}
    )
    state_user = twin_state.setdefault("state", {}).setdefault("user", {})
    intent_state = state_user.setdefault("intent", {"summary": {}, "clusters": {}, "raw": {}})
    ownership_state = state_user.setdefault("ownership", {"summary": {}, "clusters": {}, "raw": {}})
    cycle_state = state_user.setdefault("cycle_delta", {"raw": {}})
    intent_raw = intent_state.setdefault("raw", {})
    ownership_raw = ownership_state.setdefault("raw", {})
    cycle_raw = cycle_state.setdefault("raw", {})

    intent_w = {int(pid): float(_to_float(weight, 0.0)) for pid, weight in intent_raw.get("intent_weights", {}).items() if _to_float(weight, 0.0) > 0.0}
    view_w = {int(pid): float(_to_float(weight, 0.0)) for pid, weight in intent_raw.get("view_weights", {}).items() if _to_float(weight, 0.0) > 0.0}
    adj_w = {int(pid): float(_to_float(weight, 0.0)) for pid, weight in intent_raw.get("adjacent_weights", {}).items() if _to_float(weight, 0.0) > 0.0}
    cart_complement_w = {
        int(pid): float(_to_float(weight, 0.0))
        for pid, weight in intent_raw.get("cart_complement_weights", {}).items()
        if _to_float(weight, 0.0) > 0.0
    }
    for pid, weight in cart_complement_w.items():
        adj_w[pid] = float(adj_w.get(pid, 0.0) + weight)
        intent_w[pid] = float(intent_w.get(pid, 0.0) + INTENT_SOURCE_WEIGHT_MAP["adjacent"] * weight)
    purchase_w = {int(pid): float(_to_float(weight, 0.0)) for pid, weight in ownership_raw.get("purchase_weights", {}).items() if _to_float(weight, 0.0) > 0.0}
    paid_price_by_pid = {
        int(pid): (_to_float(values[0], 0.0), _to_float(values[1], 0.0))
        for pid, values in ownership_raw.get("paid_price_by_pid", {}).items()
        if isinstance(values, (list, tuple)) and len(values) == 2
    }
    recency_by_category = {}
    recency_by_subcategory = {}
    recency_by_family = {}
    recency_by_taxonomy_leaf = {}
    purchase_dates_by_category = {}
    purchase_dates_by_subcategory = {}
    purchase_dates_by_family = {}
    purchase_dates_by_taxonomy_leaf = {}
    order_dates = pd.Series(pd.to_datetime(cycle_raw.get("order_dates", []), errors="coerce", utc=True)).dropna().sort_values(kind="mergesort")
    for key, dates in cycle_raw.get("purchase_dates_by_category", {}).items():
        valid = pd.Series(pd.to_datetime(dates, errors="coerce", utc=True)).dropna().sort_values(kind="mergesort")
        if valid.empty:
            continue
        purchase_dates_by_category[key] = list(valid)
        recency_by_category[key] = int(max((now_ts - valid.max()).days, 0))
    for key, dates in cycle_raw.get("purchase_dates_by_subcategory", {}).items():
        valid = pd.Series(pd.to_datetime(dates, errors="coerce", utc=True)).dropna().sort_values(kind="mergesort")
        if valid.empty:
            continue
        purchase_dates_by_subcategory[key] = list(valid)
        recency_by_subcategory[key] = int(max((now_ts - valid.max()).days, 0))
    for key, dates in cycle_raw.get("purchase_dates_by_family", {}).items():
        valid = pd.Series(pd.to_datetime(dates, errors="coerce", utc=True)).dropna().sort_values(kind="mergesort")
        if valid.empty:
            continue
        purchase_dates_by_family[str(key)] = list(valid)
        recency_by_family[str(key)] = int(max((now_ts - valid.max()).days, 0))
    for key, dates in cycle_raw.get("purchase_dates_by_taxonomy_leaf", {}).items():
        valid = pd.Series(pd.to_datetime(dates, errors="coerce", utc=True)).dropna().sort_values(kind="mergesort")
        if valid.empty:
            continue
        purchase_dates_by_taxonomy_leaf[str(key)] = list(valid)
        recency_by_taxonomy_leaf[str(key)] = int(max((now_ts - valid.max()).days, 0))

    src_view = float(sum(view_w.values()) * INTENT_SOURCE_WEIGHT_MAP["view"])
    src_adjacent = float(sum(adj_w.values()) * INTENT_SOURCE_WEIGHT_MAP["adjacent"])
    src_total = src_view + src_adjacent
    interest_strength = float(sum(intent_w.values()))
    ownership_strength = float(sum(purchase_w.values()))
    ownership_ratio = float(ownership_strength / max(interest_strength, 1e-8))

    intent_by_family = {}
    intent_by_subcategory = {}
    intent_by_category = {}
    for pid, w in intent_w.items():
        fam = family_lookup.get(pid)
        if fam is not None and not pd.isna(fam):
            key = str(fam)
            intent_by_family[key] = float(intent_by_family.get(key, 0.0) + float(w))
        sub = subcategory_lookup.get(pid)
        if sub is not None and not pd.isna(sub):
            key = str(sub)
            intent_by_subcategory[key] = float(intent_by_subcategory.get(key, 0.0) + float(w))
        cat = main_category_lookup.get(pid)
        if cat is not None and not pd.isna(cat):
            key = str(cat)
            intent_by_category[key] = float(intent_by_category.get(key, 0.0) + float(w))

    attenuation_subcategory = 0.50
    attenuation_category = 0.25
    intent_raw["intent_weights_by_family"] = {k: float(v) for k, v in intent_by_family.items()}
    intent_raw["intent_weights_by_subcategory"] = {
        k: float(v * attenuation_subcategory) for k, v in intent_by_subcategory.items()
    }
    intent_raw["intent_weights_by_category"] = {
        k: float(v * attenuation_category) for k, v in intent_by_category.items()
    }

    brand_affinity_context = {
        "intent": _build_contextual_brand_affinity(
            weight_map=intent_w,
            brand_lookup=brand_lookup,
            family_lookup=family_lookup,
            leaf_lookup=leaf_lookup,
            main_lookup=main_category_lookup,
        ),
        "purchase": _build_contextual_brand_affinity(
            weight_map=purchase_w,
            brand_lookup=brand_lookup,
            family_lookup=family_lookup,
            leaf_lookup=leaf_lookup,
            main_lookup=main_category_lookup,
        ),
    }
    token_affinity_contexts = {}
    for attr_name, lookup, split in (
        ("size", field_lookup.get("Sizes", {}), True),
        ("color", field_lookup.get("Colors", {}), True),
        ("material", field_lookup.get("Materials", {}), True),
        ("keyword", field_lookup.get("Keywords", {}), True),
    ):
        token_affinity_contexts[attr_name] = {
            "intent": _build_contextual_token_affinity(
                weight_map=intent_w,
                value_lookup=lookup,
                family_lookup=family_lookup,
                leaf_lookup=leaf_lookup,
                main_lookup=main_category_lookup,
                split=split,
            ),
            "purchase": _build_contextual_token_affinity(
                weight_map=purchase_w,
                value_lookup=lookup,
                family_lookup=family_lookup,
                leaf_lookup=leaf_lookup,
                main_lookup=main_category_lookup,
                split=split,
            ),
        }

    raw_intent_clusters = _build_cluster_signal_blocks(
        intent_w,
        brand_lookup=brand_lookup,
        main_lookup=main_category_lookup,
        sub_lookup=subcategory_lookup,
        family_lookup=family_lookup,
        leaf_lookup=leaf_lookup,
        size_lookup=field_lookup.get("Sizes", {}),
        color_lookup=field_lookup.get("Colors", {}),
        material_lookup=field_lookup.get("Materials", {}),
        keyword_lookup=field_lookup.get("Keywords", {}),
        include_attrs=False,
    )
    raw_intent_view_clusters = _build_cluster_signal_blocks(
        view_w,
        brand_lookup=brand_lookup,
        main_lookup=main_category_lookup,
        sub_lookup=subcategory_lookup,
        family_lookup=family_lookup,
        leaf_lookup=leaf_lookup,
        size_lookup=field_lookup.get("Sizes", {}),
        color_lookup=field_lookup.get("Colors", {}),
        material_lookup=field_lookup.get("Materials", {}),
        keyword_lookup=field_lookup.get("Keywords", {}),
        include_attrs=False,
    ) if view_w else {}
    raw_intent_adjacent_clusters = _build_cluster_signal_blocks(
        adj_w,
        brand_lookup=brand_lookup,
        main_lookup=main_category_lookup,
        sub_lookup=subcategory_lookup,
        family_lookup=family_lookup,
        leaf_lookup=leaf_lookup,
        size_lookup=field_lookup.get("Sizes", {}),
        color_lookup=field_lookup.get("Colors", {}),
        material_lookup=field_lookup.get("Materials", {}),
        keyword_lookup=field_lookup.get("Keywords", {}),
        include_attrs=False,
    ) if adj_w else {}
    raw_ownership_clusters = _build_cluster_signal_blocks(
        purchase_w,
        brand_lookup=brand_lookup,
        main_lookup=main_category_lookup,
        sub_lookup=subcategory_lookup,
        family_lookup=family_lookup,
        leaf_lookup=leaf_lookup,
        size_lookup=field_lookup.get("Sizes", {}),
        color_lookup=field_lookup.get("Colors", {}),
        material_lookup=field_lookup.get("Materials", {}),
        keyword_lookup=field_lookup.get("Keywords", {}),
        include_attrs=False,
    )

    budgets = {}
    for src, key in BUDGET_ATTR_SPECS:
        lookup = field_lookup.get(src, {})
        per_val = {}
        for pid in purchase_w.keys():
            val = lookup.get(pid)
            paid_num, paid_den = paid_price_by_pid.get(pid, (0.0, 0.0))
            if paid_den > 0.0:
                price = paid_num / paid_den
            else:
                price = _to_float(price_lookup.get(pid, np.nan), np.nan)
            if val is None or np.isnan(price):
                continue
            _add_weighted_mean(per_val, val, price, purchase_w.get(pid, 1.0))
        budgets[key] = {
            attr_val: float(num / den)
            for attr_val, (num, den) in per_val.items()
            if den > 0.0
        }
    family_budget = {}
    leaf_budget = {}
    for pid in purchase_w.keys():
        paid_num, paid_den = paid_price_by_pid.get(pid, (0.0, 0.0))
        if paid_den > 0.0:
            price = paid_num / paid_den
        else:
            price = _to_float(price_lookup.get(pid, np.nan), np.nan)
        if np.isnan(price):
            continue
        fam_v = family_lookup.get(pid)
        leaf_v = leaf_lookup.get(pid)
        if fam_v is not None:
            _add_weighted_mean(family_budget, str(fam_v), price, purchase_w.get(pid, 1.0))
        if leaf_v is not None:
            _add_weighted_mean(leaf_budget, str(leaf_v), price, purchase_w.get(pid, 1.0))
    budgets["family"] = {key: float(num / den) for key, (num, den) in family_budget.items() if den > 0.0}
    budgets["taxonomy_leaf"] = {key: float(num / den) for key, (num, den) in leaf_budget.items() if den > 0.0}
    global_num = 0.0
    global_den = 0.0
    for pid, weight in purchase_w.items():
        paid_num, paid_den = paid_price_by_pid.get(pid, (0.0, 0.0))
        price = paid_num / paid_den if paid_den > 0.0 else _to_float(price_lookup.get(pid, np.nan), np.nan)
        if np.isnan(price):
            continue
        global_num += _to_float(weight, 0.0) * price
        global_den += _to_float(weight, 0.0)
    if global_den > 0.0:
        budgets["global"] = float(global_num / global_den)

    intent_clusters = _finalize_branch_clusters(
        _export_branch_clusters(
            raw_intent_clusters,
            branch_kind="intent",
            brand_context=brand_affinity_context.get("intent", {}),
            token_contexts={
                "size": token_affinity_contexts.get("size", {}).get("intent", {}),
                "color": token_affinity_contexts.get("color", {}).get("intent", {}),
                "material": token_affinity_contexts.get("material", {}).get("intent", {}),
                "keyword": token_affinity_contexts.get("keyword", {}).get("intent", {}),
            },
            intent_source_components={
                "view_engagement": _cluster_source_score_map(raw_intent_view_clusters),
                "adjacent_engagement": _cluster_source_score_map(raw_intent_adjacent_clusters),
            },
            brand_trust=ownership_raw.get("brand_trust", {}),
            brand_trust_confidence={
                brand: float(np.clip(1.0 - np.exp(-_to_float(count, 0.0) / 3.0), 0.0, 1.0))
                for brand, count in ownership_raw.get("brand_interaction_count", {}).items()
            },
            budgets=budgets,
            recency_by_category=recency_by_category,
            recency_by_subcategory=recency_by_subcategory,
        ),
        branch_kind="intent",
        budgets=budgets,
        recency_by_category=recency_by_category,
        recency_by_subcategory=recency_by_subcategory,
    )
    ownership_clusters = _finalize_branch_clusters(
        _export_branch_clusters(
            raw_ownership_clusters,
            branch_kind="ownership",
            brand_context=brand_affinity_context.get("purchase", {}),
            token_contexts={
                "size": token_affinity_contexts.get("size", {}).get("purchase", {}),
                "color": token_affinity_contexts.get("color", {}).get("purchase", {}),
                "material": token_affinity_contexts.get("material", {}).get("purchase", {}),
                "keyword": token_affinity_contexts.get("keyword", {}).get("purchase", {}),
            },
            brand_trust=ownership_raw.get("brand_trust", {}),
            brand_trust_confidence={
                brand: float(np.clip(1.0 - np.exp(-_to_float(count, 0.0) / 3.0), 0.0, 1.0))
                for brand, count in ownership_raw.get("brand_interaction_count", {}).items()
            },
            budgets=budgets,
            recency_by_category=recency_by_category,
            recency_by_subcategory=recency_by_subcategory,
        ),
        branch_kind="ownership",
        budgets=budgets,
        recency_by_category=recency_by_category,
        recency_by_subcategory=recency_by_subcategory,
    )

    observed_cycle_days = (
        float(np.median(order_dates.diff().dt.days.dropna()[lambda s: s >= 0]))
        if len(order_dates) >= 2
        else (float(max(int((order_dates.max() - order_dates.min()).days), 30)) if len(order_dates) == 1 else 90.0)
    )
    observed_cycle_days = float(np.clip(_to_float(observed_cycle_days, 90.0), 14.0, 365.0))
    days_since_last_purchase = int(max((now_ts - order_dates.max()).days, 0)) if not order_dates.empty else -1
    observed_cycle_by_category = _median_gap_lookup(purchase_dates_by_category)
    observed_cycle_by_subcategory = _median_gap_lookup(purchase_dates_by_subcategory)
    observed_cycle_by_family = _median_gap_lookup(purchase_dates_by_family)
    observed_cycle_by_taxonomy_leaf = _median_gap_lookup(purchase_dates_by_taxonomy_leaf)
    cycle_delta_ratio_by_category, faster_by_category, slower_by_category = _cycle_delta_maps(
        observed_cycle_by_category,
        category_profiles,
    )
    cycle_delta_ratio_by_subcategory, faster_by_subcategory, slower_by_subcategory = _cycle_delta_maps(
        observed_cycle_by_subcategory,
        subcategory_profiles,
    )

    user_payload = {
        "intent": {
            "summary": _build_intent_summary(
                interest_strength=interest_strength,
                view_pct=float(src_view / max(src_total, 1e-8)),
                adjacent_pct=float(src_adjacent / max(src_total, 1e-8)),
                intent_clusters=intent_clusters,
                intent_raw=intent_raw,
            ),
            "clusters": intent_clusters,
            "raw": intent_raw,
        },
        "ownership": {
            "summary": _build_ownership_summary(
                ownership_strength=ownership_strength,
                ownership_ratio=ownership_ratio,
                ownership_clusters=ownership_clusters,
                budgets=budgets,
            ),
            "clusters": ownership_clusters,
            "raw": ownership_raw,
        },
        "cycle_delta": {
            "observed_cycle_days": observed_cycle_days,
            "observed_cycle_days_by_category": observed_cycle_by_category,
            "observed_cycle_days_by_subcategory": observed_cycle_by_subcategory,
            "observed_cycle_days_by_family": observed_cycle_by_family,
            "observed_cycle_days_by_taxonomy_leaf": observed_cycle_by_taxonomy_leaf,
            "purchase_interval_by_family": observed_cycle_by_family,
            "purchase_interval_by_taxonomy_leaf": observed_cycle_by_taxonomy_leaf,
            "cycle_delta_ratio_by_category": cycle_delta_ratio_by_category,
            "cycle_delta_ratio_by_subcategory": cycle_delta_ratio_by_subcategory,
            "user_faster_than_expected_by_category": faster_by_category,
            "user_faster_than_expected_by_subcategory": faster_by_subcategory,
            "user_slower_than_expected_by_category": slower_by_category,
            "user_slower_than_expected_by_subcategory": slower_by_subcategory,
            "days_since_last_purchase": days_since_last_purchase,
            "purchase_recency_by_category": recency_by_category,
            "purchase_recency_by_subcategory": recency_by_subcategory,
            "purchase_recency_by_family": recency_by_family,
            "purchase_recency_by_taxonomy_leaf": recency_by_taxonomy_leaf,
            "budgets": budgets,
            "raw": cycle_raw,
        },
    }
    twin_state.setdefault("state", {})["user"] = user_payload

    collaborative_group_ids = collaborative_group_ids if isinstance(collaborative_group_ids, dict) else {}
    collaborative_maps_by_channel = collaborative_maps_by_channel if isinstance(collaborative_maps_by_channel, dict) else {}
    existing_collaborative = twin_state.get("collaborative_context", {}) if isinstance(twin_state.get("collaborative_context", {}), dict) else {}
    existing_channels = existing_collaborative.get("channels", {}) if isinstance(existing_collaborative.get("channels", {}), dict) else {}
    channels = {}
    if not materialize_collaborative_channels:
        channels = {}
    elif collaborative_maps_by_channel:
        for channel in COLLAB_BASE_CHANNELS:
            gid = collaborative_group_ids.get(channel)
            channel_map = collaborative_maps_by_channel.get(channel, {}).get(gid) if gid is not None else None
            if not isinstance(channel_map, dict):
                channel_map = {}
            raw_channel_clusters = _build_cluster_signal_blocks(
                channel_map,
                brand_lookup=brand_lookup,
                main_lookup=main_category_lookup,
                sub_lookup=subcategory_lookup,
                family_lookup=family_lookup,
                leaf_lookup=leaf_lookup,
                size_lookup=field_lookup.get("Sizes", {}),
                color_lookup=field_lookup.get("Colors", {}),
                material_lookup=field_lookup.get("Materials", {}),
                keyword_lookup=field_lookup.get("Keywords", {}),
            ) if channel_map else {}
            finalized_clusters = _finalize_branch_clusters(
                raw_channel_clusters,
                branch_kind="collaborative",
                budgets=budgets,
                recency_by_category=None,
                recency_by_subcategory=None,
            )
            channels[channel] = {
                "summary": _derive_collaborative_channel_support(finalized_clusters),
                "clusters": finalized_clusters,
            }
    else:
        channels = existing_channels

    twin_state = set_collaborative_context(
        twin_state,
        group_ids=collaborative_group_ids,
        channels=channels,
        updated_at=now_ts,
        provenance="collaborative_refresh",
    )
    twin_state.setdefault("state", {}).setdefault("meta", {})["updated_at"] = utc_now_iso(now_ts)
    twin_state["state"]["meta"]["freshness"] = "fresh"
    twin_state["state"]["meta"]["confidence"] = 1.0
    twin_state["state"]["meta"]["provenance"] = "event_replay"
    return twin_state

def _compute_user_signal(
    cid, now_ts, field_lookup, price_lookup, complements, substitutes, order_summary_by_client, order_item_records_by_client, view_records_by_client,
    product_id_set=None,
    collab_user_groups=None, collaborative_maps_by_channel=None, replenishment_profiles=None,
    materialize_collaborative_channels=True,
):
    collab_user_groups = collab_user_groups if isinstance(collab_user_groups, dict) else {}
    collaborative_maps_by_channel = collaborative_maps_by_channel if isinstance(collaborative_maps_by_channel, dict) else {}
    channel_group_ids = {}
    for channel, user_groups in collab_user_groups.items():
        gid = (user_groups or {}).get(cid)
        if gid is not None:
            channel_group_ids[str(channel)] = str(gid)

    transition_ctx = {
        "field_lookup": field_lookup,
        "price_lookup": price_lookup,
        "complements": complements,
        "substitutes": substitutes,
        "product_id_set": list(product_id_set or []),
        "replenishment_profiles": copy.deepcopy(replenishment_profiles or {}),
    }
    twin_state = make_empty_user_twin_state(
        user_id=cid,
        updated_at=now_ts,
        provenance="historical_replay",
    )
    user_views = view_records_by_client.get(cid)
    user_items = order_item_records_by_client.get(cid, [])
    history_events = _build_history_user_events(
        cid=cid,
        now_ts=now_ts,
        user_items=user_items,
        user_views=user_views,
        complements=complements,
        product_id_set=product_id_set,
    )
    for event in history_events:
        twin_state = _apply_user_event_to_twin_state(twin_state, {"relations": {"complements": {str(int(pid)): rels for pid, rels in complements.items()}}}, event, transition_ctx)

    twin_state = _refresh_user_twin_state(
        twin_state,
        now_ts=now_ts,
        transition_ctx=transition_ctx,
        collaborative_group_ids=channel_group_ids,
        collaborative_maps_by_channel=collaborative_maps_by_channel,
        materialize_collaborative_channels=materialize_collaborative_channels,
    )
    twin_state = set_catalog_context_ref(
        twin_state,
        {
            "kind": "catalog_context",
            "updated_at": utc_now_iso(now_ts),
        },
    )
    return twin_state

def preprocess_data_generic(
    clients, orders, order_items, products, views=None, price_history=None,
    product_relations=None, user_cols=None, item_cols=None, user_group_cols=None, num_workers: int = 1,
    progress_callback=None, verbose_stdout: bool = True, materialize_collaborative_channels: bool = True):
    clients = clients.copy()
    products = products.copy()
    orders = orders.copy()
    order_items = order_items.copy()
    if "ReviewText" not in order_items.columns and "Review" in order_items.columns:
        order_items["ReviewText"] = order_items["Review"]
    orders_signal = orders.copy()

    if "OrderStatus" in orders_signal.columns:
        orders_signal["OrderStatus"] = orders_signal["OrderStatus"].fillna("delivered").astype(str).apply(_normalize_order_status)

    products_signal = products.copy()
    products_signal.set_index("ProductID", drop=False, inplace=True)

    views_df = views.copy() if views is not None else pd.DataFrame()
    price_hist_df = _load_optional_frame(price_history, "data/price_history.csv")
    relations_df = _load_optional_frame(product_relations, "data/product_relations.csv")

    clients.set_index("ClientID", drop=False, inplace=True)
    products.set_index("ProductID", drop=False, inplace=True)

    if "OrderDate" in orders_signal.columns:
        orders_signal["OrderDate"] = pd.to_datetime(orders_signal["OrderDate"], errors="coerce", utc=True)

    if not orders_signal.empty and not order_items.empty:
        merge_cols = [c for c in ["OrderID", "ClientID", "OrderDate", "OrderStatus"] if c in orders_signal.columns]
        orders_merged = order_items.merge(orders_signal[merge_cols], on="OrderID", how="left")

        # Build per-client order summaries.
        order_summary_by_client = {}
        has_date = "OrderDate" in orders_signal.columns
        for row in orders_signal.itertuples(index=False):
            cid = int(getattr(row, "ClientID"))
            entry = order_summary_by_client.setdefault(cid, {"total_orders": 0, "order_dates": []})
            entry["total_orders"] += 1
            if has_date:
                dt = getattr(row, "OrderDate", pd.NaT)
                if pd.notna(dt):
                    entry["order_dates"].append(dt)
        for entry in order_summary_by_client.values():
            if entry["order_dates"]:
                entry["order_dates"].sort()

        # Build per-client order item records.
        order_item_records_by_client = {}
        if not orders_merged.empty and "ClientID" in orders_merged.columns:
            work = orders_merged.copy()
            if "OrderDate" in work.columns:
                work["OrderDate"] = pd.to_datetime(work["OrderDate"], errors="coerce", utc=True)
                work = work.sort_values("OrderDate", kind="mergesort")
            for row in work.itertuples(index=False):
                cid = int(getattr(row, "ClientID"))
                order_item_records_by_client.setdefault(cid, []).append(
                    (
                        int(getattr(row, "ProductID")),
                        getattr(row, "OrderStatus", "delivered"),
                        _to_float(getattr(row, "Quantity", 1.0), 1.0),
                        getattr(row, "Rating", np.nan),
                        _to_float(getattr(row, "UnitPrice", np.nan), np.nan),
                        getattr(row, "OrderDate", pd.NaT),
                    )
                )
    else:
        orders_merged = pd.DataFrame()
        order_summary_by_client = {}
        order_item_records_by_client = {}

    # Prepare view signals and exploded product IDs.
    if views_df.empty or "ProductIDs" not in views_df.columns:
        views_sig = pd.DataFrame()
        views_exploded = pd.DataFrame()
    else:
        views_sig = views_df.copy()
        views_sig["ViewDate"] = pd.to_datetime(views_sig["ViewDate"], errors="coerce", utc=True)
        views_sig["ProductIDs"] = views_sig["ProductIDs"].apply(
            lambda x: [int(pid.strip()) for pid in str(x).split(",") if pid.strip().isdigit()]
        )
        views_exploded = views_sig[["ClientID", "ProductIDs"]].explode("ProductIDs")
        views_exploded = views_exploded.dropna(subset=["ProductIDs"]).copy()
        if not views_exploded.empty:
            views_exploded["ProductID"] = views_exploded["ProductIDs"].astype(int)

    view_records_by_client = {}
    if not views_sig.empty and "ClientID" in views_sig.columns:
        for row in views_sig.itertuples(index=False):
            cid = int(getattr(row, "ClientID"))
            dt = getattr(row, "ViewDate", pd.NaT)
            pids = getattr(row, "ProductIDs", [])
            if not isinstance(pids, list):
                pids = []
            entry = view_records_by_client.setdefault(cid, {"rows": [], "last_view": pd.NaT})
            entry["rows"].append((dt, pids))
            if pd.notna(dt) and (pd.isna(entry["last_view"]) or dt > entry["last_view"]):
                entry["last_view"] = dt

    # Extract latest effective price from history, fallback to product price map.
    latest_effective_price = {}
    if not price_hist_df.empty and {"ProductID", "Date"}.issubset(price_hist_df.columns):
        ph = price_hist_df.copy()
        ph["Date"] = pd.to_datetime(ph["Date"], errors="coerce", utc=True)
        latest = ph.sort_values("Date").groupby("ProductID").tail(1)
        if "EffectivePrice" in latest.columns:
            latest_effective_price = {int(k): float(v) for k, v in zip(latest["ProductID"], latest["EffectivePrice"]) }
    if not latest_effective_price and "Price" in products_signal.columns:
        latest_effective_price = {int(k): float(v) for k, v in products_signal["Price"].to_dict().items()}

    # Build product relation maps directly.
    complements = {}
    substitutes = {}
    if not relations_df.empty and {"ProductID", "RelationType", "RelatedProductID"}.issubset(relations_df.columns):
        work = relations_df.copy()
        if "Strength" not in work.columns:
            work["Strength"] = 1.0
        for row in work.itertuples(index=False):
            pid = int(getattr(row, "ProductID"))
            rid = int(getattr(row, "RelatedProductID"))
            strength = float(getattr(row, "Strength", 1.0))
            rtype = str(getattr(row, "RelationType", "")).strip().lower()
            if "complement" in rtype:
                complements.setdefault(pid, []).append((rid, strength))
            elif "substit" in rtype:
                substitutes.setdefault(pid, []).append((rid, strength))
    prod_analytics = _build_runtime_product_analytics(
        products=products,
        orders_merged=orders_merged,
        views_exploded=views_exploded,
        price_hist_df=price_hist_df,
    )
    for col in prod_analytics.columns:
        default_value = float(RUNTIME_ANALYTICS_DEFAULTS.get(col, 0.0))
        numeric_series = pd.to_numeric(prod_analytics[col], errors="coerce").fillna(default_value).astype(float)
        products[col] = numeric_series
        products_signal[col] = numeric_series

    bought_together = _build_bought_together_map(orders_merged)
    family_lookup, family_metadata, taxonomy_leaf_lookup, family_relations = _build_family_artifacts(
        products_signal,
        complements=complements,
        bought_together=bought_together,
    )
    if family_lookup:
        products["FamilyID"] = products.index.map(lambda pid: family_lookup.get(int(pid)))
        products_signal["FamilyID"] = products_signal.index.map(lambda pid: family_lookup.get(int(pid)))
    if taxonomy_leaf_lookup:
        products["TaxonomyLeafKey"] = products.index.map(lambda pid: taxonomy_leaf_lookup.get(int(pid)))
        products_signal["TaxonomyLeafKey"] = products_signal.index.map(lambda pid: taxonomy_leaf_lookup.get(int(pid)))

    signal_fields = ["Category", "Subcategory", "Brand", "Sizes", "Colors", "Materials", "Keywords", "FamilyID", "TaxonomyLeafKey"]
    field_lookup = {f: products_signal[f].to_dict() for f in signal_fields if f in products_signal.columns}
    if "EffectivePrice" in products_signal.columns and "Price" in products_signal.columns:
        eff = products_signal["EffectivePrice"].replace(0.0, np.nan).fillna(products_signal["Price"])
        price_lookup = eff.to_dict()
    elif "Price" in products_signal.columns:
        price_lookup = products_signal["Price"].to_dict()
    else:
        price_lookup = latest_effective_price
    client_ids = clients["ClientID"].tolist()
    product_id_set = products.index.tolist()
    user_group_cols = user_group_cols or ["Age", "Gender", "Location"]
    user_group_cols = [c for c in user_group_cols if c in clients.columns]
    replenishment_profiles = _build_replenishment_profiles(orders_merged, products_signal)

    channel_user_groups, channel_group_members = _build_collaborative_group_channels(
        clients=clients,
        user_group_cols=user_group_cols,
    )
    collaborative_maps_by_channel = {}
    if materialize_collaborative_channels:
        for channel, client_to_group in channel_user_groups.items():
            channel_affinity = _build_item_group_affinity(
                orders_merged=orders_merged,
                client_to_group=client_to_group,
            )
            collaborative_maps_by_channel[channel] = _build_group_collaborative_maps(
                item_group_affinity=channel_affinity,
                group_ids=channel_group_members.get(channel, {}).keys(),
            )
    signals = {}
    now_ts = to_utc_timestamp()
    total_clients = len(client_ids)
    processed_clients = 0
    workers = max(1, int(num_workers))

    def _emit_progress(processed: int) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": "preprocess_users",
                "processed_users": int(processed),
                "total_users": int(total_clients),
                "percent": float(100.0 * processed / max(total_clients, 1)),
                "workers": int(workers),
            }
        )

    def _compute_user_signal_chunk(client_chunk):
        partial = {}
        for cid in client_chunk:
            partial[cid] = _compute_user_signal(
                cid=cid,
                now_ts=now_ts,
                field_lookup=field_lookup,
                price_lookup=price_lookup,
                complements=complements,
                substitutes=substitutes,
                order_summary_by_client=order_summary_by_client,
                order_item_records_by_client=order_item_records_by_client,
                view_records_by_client=view_records_by_client,
                product_id_set=product_id_set,
                collab_user_groups=channel_user_groups,
                collaborative_maps_by_channel=collaborative_maps_by_channel,
                replenishment_profiles=replenishment_profiles,
                materialize_collaborative_channels=materialize_collaborative_channels,
            )
        return partial

    if workers == 1:
        for cid in tqdm(client_ids, desc="Preprocessing user signals", leave=False, disable=not verbose_stdout):
            signals[cid] = _compute_user_signal(
                cid=cid,
                now_ts=now_ts,
                field_lookup=field_lookup,
                price_lookup=price_lookup,
                complements=complements,
                substitutes=substitutes,
                order_summary_by_client=order_summary_by_client,
                order_item_records_by_client=order_item_records_by_client,
                view_records_by_client=view_records_by_client,
                product_id_set=product_id_set,
                collab_user_groups=channel_user_groups,
                collaborative_maps_by_channel=collaborative_maps_by_channel,
                replenishment_profiles=replenishment_profiles,
                materialize_collaborative_channels=materialize_collaborative_channels,
            )
            processed_clients += 1
            if processed_clients == 1 or processed_clients == total_clients or processed_clients % 100 == 0:
                _emit_progress(processed_clients)
    else:
        chunk_size = max(16, len(client_ids) // max(workers * 64, 1))
        client_chunks = list(_chunk_sequence(client_ids, chunk_size))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exc:
            futures = [exc.submit(_compute_user_signal_chunk, chunk) for chunk in client_chunks]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Preprocessing user signals",
                leave=False,
                disable=not verbose_stdout,
            ):
                partial = future.result()
                signals.update(partial)
                processed_clients += len(partial)
                _emit_progress(min(processed_clients, total_clients))
    substitute_neighbors = _build_substitute_neighbor_map(substitutes)
    catalog_context = _build_catalog_context(
        products_signal=products_signal,
        price_lookup=price_lookup,
        complements=complements,
        substitutes=substitutes,
        bought_together=bought_together,
        substitute_neighbors=substitute_neighbors,
        replenishment_profiles=replenishment_profiles,
        product_family_lookup=family_lookup,
        family_metadata=family_metadata,
        family_relations=family_relations,
        taxonomy_leaf_lookup=taxonomy_leaf_lookup,
    )
    signals["_catalog_context"] = catalog_context
    catalog_ref = {
        "kind": "catalog_context",
        "product_count": int(len((catalog_context.get("source_dynamics", {}) or {}).get("products", {}))),
        "updated_at": catalog_context.get("updated_at"),
    }
    for cid in client_ids:
        payload = signals.get(cid)
        if is_user_twin_state(payload):
            signals[cid] = set_catalog_context_ref(payload, catalog_ref)

    return clients, products, orders, order_items, views_df, signals
