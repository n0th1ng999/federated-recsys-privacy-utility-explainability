ATTR_SPECS = [
    ("Category", "category", False),
    ("Subcategory", "subcategory", False),
    ("Brand", "brand", False),
    ("Sizes", "size", True),
    ("Colors", "color", True),
    ("Materials", "material", True),
    ("Keywords", "keyword", True),
]

BUDGET_ATTR_SPECS = [
    ("Category", "category"),
    ("Subcategory", "subcategory"),
]

INTENT_SOURCE_SPECS = [
    ("view", 1.0),
    ("adjacent", 0.75),
]
INTENT_SOURCE_WEIGHT_MAP = dict(INTENT_SOURCE_SPECS)
COMPLEMENT_ADJACENT_WEIGHT = 0.9
COMPLEMENT_CART_ADJACENT_WEIGHT = 1.25
COMPLEMENT_BOOST_TAU_DAYS = 6.0
COMPLEMENT_BOOST_MAX_AGE_DAYS = 14.0
AGE_BUCKETS = [
    ("age_lt25", 18.0, 25.0),
    ("age_25_34", 25.0, 35.0),
    ("age_35_44", 35.0, 45.0),
    ("age_45_54", 45.0, 55.0),
    ("age_55_plus", 55.0, 70.0),
]
AGE_SMOOTH_WIDTH_YEARS = 4.0
BRAND_TRUST_DECAY = 0.88
BRAND_TRUST_MAX_ABS = 1.0
BRAND_BAD_STREAK_SUPPRESS_THRESHOLD = 2
ORDER_STATUS_MAP = {
    "delivered": "delivered",
    "entregue": "delivered",
    "returned": "returned",
    "devolvido": "returned",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "cancelado": "cancelled",
}
