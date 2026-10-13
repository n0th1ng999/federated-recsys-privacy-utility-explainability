SIGNAL_POLICIES = {
    "brand": {"intent_scope": "source_local", "purchase_scope": "global", "intent_weight": 1.0, "purchase_weight": 1.25},
    "keyword": {"intent_scope": "source_local", "purchase_scope": "source_local", "intent_weight": 1.15, "purchase_weight": 0.35},
    "size": {"intent_scope": "source_local", "purchase_scope": "parent_category", "intent_weight": 0.05, "purchase_weight": 0.75},
    "color": {"intent_scope": "source_local", "purchase_scope": "source_local", "intent_weight": 0.07, "purchase_weight": 0.35},
    "material": {"intent_scope": "source_local", "purchase_scope": "source_local", "intent_weight": 0.10, "purchase_weight": 0.45},
}

SCORE_ATTR_KEYS = tuple(SIGNAL_POLICIES.keys())
PURCHASE_WEIGHTS = {attr: cfg["purchase_weight"] for attr, cfg in SIGNAL_POLICIES.items()}
SOURCE_INTENT_WEIGHT = 2.2
SOURCE_PURCHASE_WEIGHT = 0.4
UNKNOWN_SOURCE_TOKENS = {"", "0", "unknown", "n/a", "n.a.", "na", "none", "null"}
COLLAB_BASE_CHANNELS = ("age", "gender", "location")
COLLAB_CHANNEL_SUPPORT_FEATURES = ["channel_score", "top_cluster_strength"]
COLLABORATIVE_CLUSTER_TOP_K = 4

USER_STATE_INTENT_SUMMARY_FEATURES = [
    "interest_strength",
    "source_view_share",
    "source_adjacent_share",
    "active_view_mass",
    "active_wishlist_mass",
    "active_cart_mass",
    "source_diversity",
    "focus_concentration",
]

USER_STATE_OWNERSHIP_SUMMARY_FEATURES = [
    "ownership_strength",
    "ownership_interest_ratio",
    "budget_global_norm",
    "budget_coverage",
    "ownership_diversity",
    "recency_coverage",
]

COLLABORATIVE_CHANNEL_SUPPORT_FEATURES = [
    f"{channel}_{feature}"
    for channel in COLLAB_BASE_CHANNELS
    for feature in COLLAB_CHANNEL_SUPPORT_FEATURES
]

INTENT_CLUSTER_FEATURE_NAMES = [
    "source_strength",
    "view_engagement",
    "adjacent_engagement",
    "active_cluster",
    "fulfilled_suppression",
]

OWNERSHIP_CLUSTER_FEATURE_NAMES = [
    "source_strength",
    "active_cluster",
    "budget_target_log",
]

COLLABORATIVE_CLUSTER_FEATURE_NAMES = [
    "is_age",
    "is_gender",
    "is_location",
    "channel_score",
    "source_strength",
    "active_cluster",
]

PAIR_CONTEXT_FEATURE_NAMES = [
    "intent_score",
    "source_intent_score",
    "family_local_intent",
    "source_purchase_recency",
    "source_family_recovery",
    "expected_cycle_norm",
    "user_cycle_ratio",
    "user_faster_than_expected",
    "user_slower_than_expected",
    "same_item_penalty",
    "substitute_penalty",
    "cluster_suppression",
    "brand_match",
    "keyword_match",
    "size_match",
    "color_match",
    "material_match",
    "source_purchase_score",
    "source_budget_alignment",
    "global_budget_alignment",
    "price_norm",
    "intent_momentum",
    "interaction_boost",
]

PURCHASE_REPLENISHMENT_INTENT_WEIGHTS = {
    "source_family_recovery": 0.44,
    "source_purchase_recency": 0.22,
    "expected_cycle_norm": 0.14,
    "user_faster_than_expected": 0.12,
    "user_slower_than_expected": -0.10,
    "user_cycle_ratio": 0.08,
}

PURCHASE_FAMILY_INTENT_WEIGHTS = {
    "behavioral_intent": 0.35,
    "intent_score": 0.19,
    "source_intent_score": 0.06,
    "intent_momentum": 0.08,
    "interaction_boost": 0.10,
    "replenishment_intent": 0.15,
    "same_item_relief": 0.04,
    "substitute_relief": 0.02,
    "cluster_relief": 0.01,
}

PURCHASE_PRODUCT_FIT_WEIGHTS = {
    "brand_match": 0.22,
    "keyword_match": 0.18,
    "size_match": 0.12,
    "color_match": 0.08,
    "material_match": 0.10,
    "source_purchase_score": 0.10,
    "source_budget_alignment": 0.12,
    "global_budget_alignment": 0.05,
    "price_norm": 0.03,
}

PURCHASE_LOGIT_WEIGHTS = {
    "bias": -1.62,
    "family_intent": 1.64,
    "product_fit": 1.16,
    "intent_fit_synergy": 1.84,
    "intent_fit_compound": 1.10,
}
PURCHASE_LOGIT_SCALE = 1.18

PURCHASE_PRIOR_SUPPORT_WEIGHTS = {
    "hist_support": 0.06,
    "pair_support": 0.05,
    "intent_support": 0.04,
    "ownership_support": 0.03,
}

PURCHASE_CONFIDENCE_WEIGHTS = {
    "family_intent": 0.38,
    "product_fit": 0.28,
    "intent_fit_synergy": 0.16,
    "intent_fit_compound": 0.10,
    "rank_preference": 0.08,
}
