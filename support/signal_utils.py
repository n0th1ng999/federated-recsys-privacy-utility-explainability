import ast
import re
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from support.feature_contract import UNKNOWN_SOURCE_TOKENS
from support.preprocess_contract import (
    AGE_BUCKETS,
    AGE_SMOOTH_WIDTH_YEARS,
    ORDER_STATUS_MAP,
)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        if isinstance(value, float) and np.isnan(value):
            return float(default)
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def fill_frame_numeric_nulls(frame):
    out = frame.copy()
    if out.empty:
        return out
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        out.loc[:, numeric_cols] = out.loc[:, numeric_cols].fillna(0)
    return out


def clip(value, lo=0.0, hi=1.0):
    return float(np.clip(safe_float(value, lo), lo, hi))


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-safe_float(x, 0.0))))


def exp_decay(days, tau):
    if days is None:
        return 0.0
    d = max(safe_float(days, 0.0), 0.0)
    t = max(safe_float(tau, 1.0), 1e-8)
    return float(np.exp(-d / t))


def normalize_attr_token(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    token = str(value).strip()
    return token or None


def normalize_attr_tokens(value):
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set)):
        raw_tokens = list(value)
    else:
        try:
            if pd.isna(value):
                return tuple()
        except Exception:
            pass
        raw_tokens = str(value).split(",")

    out = []
    seen = set()
    for raw in raw_tokens:
        token = normalize_attr_token(raw)
        if token is None or token in seen:
            continue
        out.append(token)
        seen.add(token)
    return tuple(out)


def cacheable_attr_value(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(sorted(value))
    return value


def is_real_source_token(value):
    token = normalize_attr_token(value)
    if token is None:
        return False
    return token.strip().lower() not in UNKNOWN_SOURCE_TOKENS


def split_product_ids(value):
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    out = []
    for token in raw_items:
        token = str(token).strip()
        if token.isdigit():
            out.append(int(token))
    return out


def to_utc_datetime(values):
    return pd.to_datetime(values, errors="coerce", utc=True)


def utc_now():
    return pd.Timestamp.now(tz="UTC")


def normalize_lookup(lookup):
    if not lookup:
        return {}
    vals = np.array(list(lookup.values()), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo <= 1e-12:
        return {k: 0.0 for k in lookup.keys()}
    return {k: float((safe_float(v, lo) - lo) / (hi - lo)) for k, v in lookup.items()}


def log_norm(value, scale):
    v = max(safe_float(value, 0.0), 0.0)
    s = max(safe_float(scale, 1.0), 1e-8)
    return clip(np.log1p(v) / np.log1p(s))


def safe_norm_entropy(values):
    if not values or len(values) <= 1:
        return 0.0
    probs = np.asarray(values, dtype=float)
    probs = probs[np.isfinite(probs)]
    if probs.size <= 1:
        return 0.0
    total = float(probs.sum())
    if total <= 1e-12:
        return 0.0
    probs = probs / total
    ent = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    denom = float(np.log(len(probs)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(ent / denom, 0.0, 1.0))


def derive_collaborative_channel_support(clusters):
    cluster_values = clusters.values() if isinstance(clusters, dict) else clusters
    strengths = []
    for cluster in cluster_values or []:
        if not isinstance(cluster, dict):
            continue
        raw_signals = cluster.get("signals", {})
        if isinstance(raw_signals, dict):
            strengths.append(float(np.clip(safe_float(raw_signals.get("source_strength", 0.0), 0.0), 0.0, 1.0)))
    strengths = [value for value in strengths if value > 0.0]
    if not strengths:
        return {
            "channel_score": 0.0,
            "top_cluster_strength": 0.0,
        }
    sorted_strengths = sorted(strengths, reverse=True)
    total_strength = float(sum(sorted_strengths))
    top_cluster_strength = float(sorted_strengths[0] / max(total_strength, 1e-8))
    support_strength = float(1.0 - np.exp(-total_strength / 2.5))
    return {
        "channel_score": float(np.clip(support_strength * top_cluster_strength, 0.0, 1.0)),
        "top_cluster_strength": top_cluster_strength,
    }


def to_list_value(value):
    if isinstance(value, (list, tuple, set)):
        return [str(tok).strip().strip("\"'") for tok in value if str(tok).strip().strip("\"'")]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return []
    if text[0] in "[({" and text[-1] in "])}":
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(tok).strip().strip("\"'") for tok in parsed if str(tok).strip().strip("\"'")]
        text = text[1:-1].strip()
    return [tok.strip().strip("\"'") for tok in text.split(",") if tok.strip().strip("\"'")]


def normalize_order_status(value):
    raw = str(value).strip().lower()
    return ORDER_STATUS_MAP.get(raw, raw)


def rating_to_sentiment(rating):
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return 0.0
    try:
        return float(np.clip((float(rating) - 3.0) / 2.0, -1.0, 1.0))
    except Exception:
        return 0.0


def normalize_group_token(value, prefix):
    if value is None or pd.isna(value):
        return f"{prefix}_unknown"
    token = str(value).strip().lower()
    return token if token else f"{prefix}_unknown"


def age_bucket(value):
    age = safe_float(value, np.nan)
    if np.isnan(age):
        return "age_unknown"
    for name, low, high in AGE_BUCKETS:
        if age >= low and age < high:
            return name
    return AGE_BUCKETS[-1][0]


def age_bucket_membership(value, smooth_width=AGE_SMOOTH_WIDTH_YEARS):
    age = safe_float(value, np.nan)
    if np.isnan(age):
        return {"age_unknown": 1.0}
    smooth_width = max(safe_float(smooth_width, 0.0), 0.0)
    idx = None
    for i, (_, low, high) in enumerate(AGE_BUCKETS):
        if age >= low and age < high:
            idx = i
            break
    if idx is None:
        idx = len(AGE_BUCKETS) - 1
    primary_name, low, high = AGE_BUCKETS[idx]
    weights = {primary_name: 1.0}
    if smooth_width <= 1e-8:
        return weights
    if np.isfinite(low) and idx > 0:
        dist_low = max(age - low, 0.0)
        if dist_low < smooth_width:
            prev_name = AGE_BUCKETS[idx - 1][0]
            weights[prev_name] = weights.get(prev_name, 0.0) + ((smooth_width - dist_low) / smooth_width)
    if np.isfinite(high) and idx < (len(AGE_BUCKETS) - 1):
        dist_high = max(high - age, 0.0)
        if dist_high < smooth_width:
            next_name = AGE_BUCKETS[idx + 1][0]
            weights[next_name] = weights.get(next_name, 0.0) + ((smooth_width - dist_high) / smooth_width)
    total = sum(max(safe_float(w, 0.0), 0.0) for w in weights.values())
    if total <= 1e-12:
        return {primary_name: 1.0}
    return {k: float(max(safe_float(v, 0.0), 0.0) / total) for k, v in weights.items()}


def add_weighted_mean(agg, key, value, weight):
    if key is None or pd.isna(key):
        return
    v = safe_float(value, np.nan)
    w = safe_float(weight, 0.0)
    if np.isnan(v) or w <= 0.0:
        return
    num, den = agg.get(key, (0.0, 0.0))
    agg[key] = (num + (v * w), den + w)


def ensure_columns(df, defaults):
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default


def norm_log_series(series):
    vals = np.log1p(np.clip(series.astype(float).values, a_min=0.0, a_max=None))
    if len(vals) == 0:
        return np.zeros(0, dtype=float)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo <= 1e-12:
        return np.zeros_like(vals, dtype=float)
    return (vals - lo) / (hi - lo)


def normalize_scalar_token(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    token = str(value).strip()
    previous = None
    while token and token != previous:
        previous = token
        if len(token) >= 2 and ((token[0], token[-1]) in {('"', '"'), ("'", "'"), ("[", "]"), ("(", ")"), ("{", "}")}):
            token = token[1:-1].strip()
    return token or None


def normalize_split_tokens(value):
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set)):
        raw_tokens = list(value)
    else:
        try:
            if pd.isna(value):
                return tuple()
        except Exception:
            pass
        raw_tokens = to_list_value(value)
    out = []
    seen = set()
    for raw in raw_tokens:
        token = normalize_scalar_token(raw)
        if token is None or token in seen:
            continue
        out.append(token)
        seen.add(token)
    return tuple(out)


def cluster_key(level, value):
    if value is None:
        return None
    return f"{level}::{value}"


def decay_signal(days, tau):
    days = safe_float(days, np.nan)
    if np.isnan(days) or days < 0:
        return 0.0
    return float(np.exp(-days / max(safe_float(tau, 1.0), 1e-8)))


def fit_tfidf_if_possible(series, max_features):
    if series is None:
        return None
    text = (
        pd.Series(series, copy=False)
        .fillna("")
        .astype(str)
        .map(lambda x: re.sub(r"\s+", " ", x).strip())
    )
    text = text[text.str.contains(r"(?u)\b\w\w+\b", regex=True, na=False)]
    if text.empty:
        return None
    vectorizer = TfidfVectorizer(max_features=max_features)
    try:
        vectorizer.fit(text)
    except ValueError:
        return None
    return vectorizer


def taxonomy_leaf_spec(category, subcategory):
    category_token = normalize_scalar_token(category)
    subcategory_token = normalize_scalar_token(subcategory)
    if is_real_source_token(subcategory_token):
        return {
            "level": "subcategory",
            "value": subcategory_token,
            "parent": category_token if is_real_source_token(category_token) else None,
            "key": cluster_key("subcategory", subcategory_token),
            "category": category_token if is_real_source_token(category_token) else None,
            "subcategory": subcategory_token,
        }
    if is_real_source_token(category_token):
        return {
            "level": "category",
            "value": category_token,
            "parent": None,
            "key": cluster_key("category", category_token),
            "category": category_token,
            "subcategory": None,
        }
    return {
        "level": "unknown",
        "value": None,
        "parent": None,
        "key": None,
        "category": category_token if is_real_source_token(category_token) else None,
        "subcategory": subcategory_token if is_real_source_token(subcategory_token) else None,
    }


def family_vector_text(row):
    parts = []
    for label, columns in (
        ("brand", ("Brand",)),
        ("size", ("Sizes", "Size")),
        ("color", ("Colors", "Color")),
        ("material", ("Materials", "Material")),
    ):
        value = None
        for column in columns:
            if column in row.index:
                value = row.get(column)
                break
        if value is None:
            continue
        tokens = to_list_value(value) if label != "brand" else [normalize_scalar_token(value)]
        tokens = [str(tok).strip() for tok in tokens if str(tok).strip()]
        if tokens:
            parts.append(f"{label} " + " ".join(tokens))
    return " ".join(parts)


def family_numeric_matrix(bucket_df):
    numeric_cols = []
    for column in bucket_df.columns:
        if column == "ProductID":
            continue
        if pd.api.types.is_numeric_dtype(bucket_df[column]):
            numeric_cols.append(column)
    if not numeric_cols:
        return np.zeros((len(bucket_df), 1), dtype=np.float32)
    matrix = (
        bucket_df[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if matrix.size == 0:
        return np.zeros((len(bucket_df), 1), dtype=np.float32)
    col_mean = matrix.mean(axis=0, keepdims=True)
    col_std = matrix.std(axis=0, keepdims=True)
    col_std[col_std < 1e-6] = 1.0
    return ((matrix - col_mean) / col_std).astype(np.float32)


def family_text_matrix(series, max_features):
    vectorizer = fit_tfidf_if_possible(series, max_features=max_features)
    if vectorizer is None:
        return np.zeros((len(series), 1), dtype=np.float32)
    matrix = vectorizer.transform(pd.Series(series, copy=False).fillna("").astype(str))
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def l2_normalize_rows(matrix):
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return (arr / norms).astype(np.float32)


def family_neighbor_search(vectors, top_k):
    arr = np.asarray(vectors, dtype=np.float32)
    n_rows = int(arr.shape[0]) if arr.ndim == 2 else 0
    if n_rows == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int64)
    k = min(max(2, int(top_k)), n_rows)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    sims, idxs = index.search(arr, k)
    return sims.astype(np.float32), idxs.astype(np.int64)


def accumulate_token_weight(target, value, weight):
    token = normalize_attr_token(value)
    if token is None:
        return
    target[token] += max(safe_float(weight, 0.0), 0.0)


def normalize_weight_map(raw):
    if not isinstance(raw, dict) or not raw:
        return {}
    max_value = max((safe_float(v, 0.0) for v in raw.values()), default=0.0)
    if max_value <= 0.0:
        return {}
    return {str(k): clip(safe_float(v, 0.0) / max_value) for k, v in raw.items() if safe_float(v, 0.0) > 0.0}


def best_token_match(values, score_map):
    if not isinstance(score_map, dict) or not score_map:
        return 0.0
    if values is None:
        return 0.0
    if not isinstance(values, (tuple, list, set)):
        values = [values]
    best = 0.0
    for value in values:
        token = normalize_attr_token(value)
        if token is None:
            continue
        score = clip(score_map.get(token, 0.0))
        if score > best:
            best = score
    return best
