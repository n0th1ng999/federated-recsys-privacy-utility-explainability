from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


TWIN_STATE_SCHEMA_VERSION = 2


def to_utc_timestamp(value: Any = None) -> pd.Timestamp:
    # RECSYS_REFERENCE_TIME pins "now" for offline experiments (temporal holdout cutoff),
    # so recency features do not depend on the wall-clock date of the run.
    if value is None:
        value = os.environ.get("RECSYS_REFERENCE_TIME") or None
    ts = pd.Timestamp.utcnow() if value is None else pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def utc_now_iso(value: Any = None) -> str:
    return to_utc_timestamp(value).isoformat()


def _event_envelope(
    kind: str,
    *,
    occurred_at: Any = None,
    origin: str = "observed",
    user_id: Optional[int] = None,
    product_id: Optional[int] = None,
    source_refs: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event = {
        "kind": str(kind),
        "occurred_at": utc_now_iso(occurred_at),
        "origin": str(origin),
        "payload": dict(payload or {}),
    }
    if user_id is not None:
        event["user_id"] = int(user_id)
    if product_id is not None:
        event["product_id"] = int(product_id)
    if source_refs:
        event["source_refs"] = dict(source_refs)
    return event


def build_user_event(
    kind: str,
    *,
    occurred_at: Any = None,
    origin: str = "observed",
    user_id: Optional[int] = None,
    product_id: Optional[int] = None,
    source_refs: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event = _event_envelope(
        kind,
        occurred_at=occurred_at,
        origin=origin,
        user_id=user_id,
        product_id=product_id,
        source_refs=source_refs,
        payload=payload,
    )
    event["family"] = "user"
    return event


def build_catalog_event(
    kind: str,
    *,
    occurred_at: Any = None,
    origin: str = "observed",
    product_id: Optional[int] = None,
    source_refs: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event = _event_envelope(
        kind,
        occurred_at=occurred_at,
        origin=origin,
        product_id=product_id,
        source_refs=source_refs,
        payload=payload,
    )
    event["family"] = "catalog"
    return event


def online_intent_to_user_events(
    online_intent: Optional[Dict[str, Any]],
    *,
    occurred_at: Any = None,
    origin: str = "session",
    user_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(online_intent, dict):
        return []

    def _int_list(values: Any) -> List[int]:
        out: List[int] = []
        seen = set()
        if not isinstance(values, (list, tuple, set)):
            return out
        for raw_value in values:
            try:
                value = int(raw_value)
            except Exception:
                continue
            if value <= 0 or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    events: List[Dict[str, Any]] = []
    for kind, field_name in (
        ("view", "viewed_product_ids"),
        ("wishlist_add", "wishlist_product_ids"),
        ("cart_add", "cart_product_ids"),
    ):
        product_ids = _int_list(online_intent.get(field_name, []))
        if not product_ids:
            continue
        events.append(
            build_user_event(
                kind,
                occurred_at=occurred_at,
                origin=origin,
                user_id=user_id,
                payload={"product_ids": product_ids},
            )
        )
    return events


def make_empty_user_twin_state(
    *,
    user_id: Optional[int] = None,
    updated_at: Any = None,
    provenance: str = "historical_replay",
    freshness: str = "fresh",
    confidence: float = 1.0,
) -> Dict[str, Any]:
    return {
        "schema_version": TWIN_STATE_SCHEMA_VERSION,
        "mode": "user_twin_state",
        "entity": {"user_id": int(user_id) if user_id is not None else None},
        "state": {
            "user": {
                "intent": {
                    "summary": {},
                    "clusters": {},
                    "raw": {
                        "intent_weights": {},
                        "view_weights": {},
                        "adjacent_weights": {},
                        "wishlist_weights": {},
                        "cart_weights": {},
                        "active_view_product_ids": [],
                        "active_wishlist_product_ids": [],
                        "active_cart_product_ids": [],
                    },
                },
                "ownership": {
                    "summary": {},
                    "clusters": {},
                    "raw": {
                        "purchase_weights": {},
                        "paid_price_by_pid": {},
                        "brand_trust": {},
                        "brand_bad_streak": {},
                        "brand_interaction_count": {},
                    },
                },
                "cycle_delta": {
                    "observed_cycle_days": 90.0,
                    "observed_cycle_days_by_category": {},
                    "observed_cycle_days_by_subcategory": {},
                    "cycle_delta_ratio_by_category": {},
                    "cycle_delta_ratio_by_subcategory": {},
                    "user_faster_than_expected_by_category": {},
                    "user_faster_than_expected_by_subcategory": {},
                    "user_slower_than_expected_by_category": {},
                    "user_slower_than_expected_by_subcategory": {},
                    "days_since_last_purchase": -1,
                    "purchase_recency_by_category": {},
                    "purchase_recency_by_subcategory": {},
                    "budgets": {},
                    "raw": {
                        "purchase_dates_by_category": {},
                        "purchase_dates_by_subcategory": {},
                        "order_dates": [],
                    },
                },
            },
            "meta": {
                "updated_at": utc_now_iso(updated_at),
                "freshness": str(freshness),
                "confidence": float(confidence),
                "provenance": str(provenance),
            },
        },
        "collaborative_context": {
            "group_ids": {},
            "channels": {},
            "meta": {
                "updated_at": utc_now_iso(updated_at),
                "freshness": str(freshness),
                "confidence": float(confidence),
                "provenance": "collaborative_refresh",
            },
        },
        "catalog_context_ref": None,
    }


def is_user_twin_state(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("mode") == "user_twin_state"
    )


def clone_user_twin_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(payload if isinstance(payload, dict) else {})


def set_collaborative_context(
    payload: Dict[str, Any],
    *,
    group_ids: Optional[Dict[str, Any]] = None,
    channels: Optional[Dict[str, Any]] = None,
    updated_at: Any = None,
    freshness: str = "fresh",
    confidence: float = 1.0,
    provenance: str = "collaborative_refresh",
) -> Dict[str, Any]:
    out = clone_user_twin_state(payload)
    collab = out.setdefault("collaborative_context", {})
    collab["group_ids"] = {
        str(channel): str(group_id)
        for channel, group_id in (group_ids or {}).items()
        if group_id is not None
    }
    collab["channels"] = copy.deepcopy(channels or {})
    collab["meta"] = {
        "updated_at": utc_now_iso(updated_at),
        "freshness": str(freshness),
        "confidence": float(confidence),
        "provenance": str(provenance),
    }
    return out


def set_catalog_context_ref(
    payload: Dict[str, Any],
    catalog_context_ref: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out = clone_user_twin_state(payload)
    out["catalog_context_ref"] = copy.deepcopy(catalog_context_ref) if isinstance(catalog_context_ref, dict) else None
    return out


def extract_user_domain(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not is_user_twin_state(payload):
        return {}
    state = payload.get("state", {})
    if not isinstance(state, dict):
        return {}
    user = state.get("user", {})
    return user if isinstance(user, dict) else {}


def extract_collaborative_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not is_user_twin_state(payload):
        return {}
    context = payload.get("collaborative_context", {})
    return context if isinstance(context, dict) else {}


def iter_active_product_ids(raw_block: Any) -> Iterable[int]:
    if not isinstance(raw_block, dict):
        return ()
    merged = []
    for key in ("active_view_product_ids", "active_wishlist_product_ids", "active_cart_product_ids"):
        values = raw_block.get(key, [])
        if not isinstance(values, list):
            continue
        for raw_value in values:
            try:
                merged.append(int(raw_value))
            except Exception:
                continue
    return merged


def clone_catalog_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(payload if isinstance(payload, dict) else {})


def apply_catalog_event(
    catalog_context: Optional[Dict[str, Any]],
    event: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    context = clone_catalog_context(catalog_context or {})
    if not isinstance(event, dict) or str(event.get("family") or "") != "catalog":
        return context

    payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
    kind = str(event.get("kind") or "").strip().lower()
    product_id = event.get("product_id")
    try:
        product_key = str(int(product_id)) if product_id is not None else None
    except Exception:
        product_key = None

    relations = context.setdefault("relations", {})
    source_dynamics = context.setdefault("source_dynamics", {})
    products = source_dynamics.setdefault("products", {})
    meta = context.setdefault("meta", {})
    meta["updated_at"] = utc_now_iso(event.get("occurred_at"))
    meta["freshness"] = "live"
    meta["provenance"] = "catalog_event"

    if kind == "price_change" and product_key is not None:
        product_entry = products.setdefault(product_key, {"product_id": int(product_key)})
        try:
            price_value = float(payload.get("price"))
        except Exception:
            price_value = None
        if price_value is not None:
            product_entry["price"] = price_value
    elif kind == "stock_change" and product_key is not None:
        product_entry = products.setdefault(product_key, {"product_id": int(product_key)})
        stock_block = product_entry.setdefault("stock", {})
        if "quantity" in payload:
            try:
                stock_block["quantity"] = float(payload.get("quantity"))
            except Exception:
                pass
        if "status" in payload:
            stock_block["status"] = payload.get("status")
        if "is_available" in payload:
            stock_block["is_available"] = payload.get("is_available")
    elif kind == "visibility_change" and product_key is not None:
        product_entry = products.setdefault(product_key, {"product_id": int(product_key)})
        visibility_block = product_entry.setdefault("visibility", {})
        for key in ("post_status", "catalog_visibility", "serving_visible"):
            if key in payload:
                visibility_block[key] = payload.get(key)
    elif kind == "relation_change":
        relation_type = str(payload.get("relation_type") or "").strip().lower()
        if relation_type not in {"complements", "substitutes", "substitute_neighbors", "bought_together"}:
            return context
        related_product_id = payload.get("related_product_id")
        try:
            related_key = str(int(related_product_id)) if related_product_id is not None else None
        except Exception:
            related_key = None
        if product_key is None or related_key is None:
            return context
        op = str(payload.get("op") or "upsert").strip().lower()
        relation_block = relations.setdefault(relation_type, {})
        strength = payload.get("strength", 0.0)
        try:
            strength_value = float(strength)
        except Exception:
            strength_value = 0.0

        if relation_type in {"substitute_neighbors", "bought_together"}:
            rel_map = relation_block.setdefault(product_key, {})
            if op == "remove":
                rel_map.pop(related_key, None)
            else:
                rel_map[related_key] = strength_value
        else:
            rel_list = relation_block.setdefault(product_key, [])
            normalized = []
            removed = False
            for row in rel_list:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                try:
                    row_pid = int(row[0])
                except Exception:
                    continue
                if str(row_pid) == related_key:
                    removed = True
                    if op == "remove":
                        continue
                    normalized.append((row_pid, strength_value))
                else:
                    normalized.append((row_pid, float(row[1])))
            if op != "remove" and not removed:
                normalized.append((int(related_key), strength_value))
            relation_block[product_key] = normalized

    return context
