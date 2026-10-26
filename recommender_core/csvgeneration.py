import os
import random
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

fake = Faker()

# -------------------------------
# PARAMETERS
# -------------------------------
NUM_CLIENTS = 9000
NUM_ITEMS = 2400
NUM_LOCATIONS = 20
MAX_SESSIONS = 20
MAX_VIEWS_PER_SESSION = 40
ANON_RATIO = 0.15

SEARCH_TERMS = [
    "budget", "premium", "eco", "durable", "lightweight", "gift", "home", "office", "daily", "new"
]
SORT_OPTIONS = ["relevance", "price_asc", "price_desc", "rating_desc", "newest"]
PROMO_TYPES = ["none", "flash_sale", "coupon", "bundle", "seasonal"]
RETURN_REASONS = ["damaged", "not_as_described", "size_issue", "late_delivery", "changed_mind"]
REVIEW_SNIPPETS = [
    "Good quality for the price",
    "Exactly what I expected",
    "Average overall",
    "Build quality could be better",
    "Would buy again",
    "Not a good fit for my needs",
    "Great value",
]


# -------------------------------
# CLIENT GENERATION
# -------------------------------
def generate_clients(num_clients):
    clients = []
    used_names = set()
    for i in tqdm(range(num_clients), desc="Generating clients"):
        age = int(np.clip(np.random.normal(35, 12), 18, 70))
        gender = np.random.choice(["M", "F", "O"], p=[0.49, 0.49, 0.02])
        while True:
            first = fake.first_name_male() if gender == "M" else fake.first_name_female() if gender == "F" else fake.first_name()
            last = fake.last_name()
            name = f"{first} {last}"
            if name not in used_names:
                used_names.add(name)
                break
        email = f"user{i}@example.com"
        location = f"Loc_{np.random.randint(1, NUM_LOCATIONS + 1)}"
        clients.append([i + 1, name, email, age, gender, location])
    return pd.DataFrame(clients, columns=["ClientID", "Name", "Email", "Age", "Gender", "Location"])


def generate_items(num_items):
    categories = [f"Category_{i}" for i in range(1, 9)]
    subcategories = [f"Subcategory_{i}" for i in range(1, 21)]
    brands = [f"Brand_{i}" for i in range(1, 41)]

    cat_to_subs = {cat: [] for cat in categories}
    for idx, sub in enumerate(subcategories):
        cat_to_subs[categories[idx % len(categories)]].append(sub)

    global_brand_count = max(4, len(brands) // 10)
    global_brand_pool = list(np.random.choice(brands, size=global_brand_count, replace=False))
    non_global_brands = [b for b in brands if b not in set(global_brand_pool)]

    cat_brand_count = max(6, len(brands) // len(categories))
    cat_to_brands = {}
    for cat in categories:
        cat_specific = list(np.random.choice(non_global_brands, size=cat_brand_count, replace=False))
        cat_to_brands[cat] = global_brand_pool + cat_specific

    colors_placeholder = [
        "Red", "Blue", "Green", "Black", "White",
        "Yellow", "Purple", "Orange", "Grey", "Brown"
    ]

    sizes_placeholder = [
        "XS", "S", "M", "L", "XL", "XXL", "XXXL"
    ]

    materials_placeholder = [
        "Cotton", "Polyester", "Plastic", "Metal", "Wood",
        "Glass", "Rubber", "Leather"
    ]

    certifications_placeholder = [
        "Cert_A", "Cert_B", "Cert_C", "Cert_D", "Cert_E", "Cert_F"
    ]

    keywords_placeholder = [
        "generic", "popular", "sale", "trending", "novel",
        "eco", "durable", "lightweight", "premium", "budget"
    ]

    items = []
    for i in tqdm(range(num_items), desc="Generating products"):
        product_id = i + 1
        name = f"Item_{product_id}"

        category = np.random.choice(categories)

        if np.random.random() < 0.93:
            subcategory = np.random.choice(cat_to_subs[category])
        else:
            subcategory = np.random.choice(subcategories)

        # Tightened distributions
        price = round(np.clip(np.random.normal(120, 60), 5, 500), 2)
        stock = int(np.clip(np.random.exponential(scale=120), 0, 500))
        created_at = datetime.utcnow() - timedelta(days=int(np.random.exponential(scale=400)))

        is_new = np.random.random() < 0.2

        if np.random.random() < 0.9:
            brand = np.random.choice(cat_to_brands[category])
        else:
            brand = np.random.choice(brands)

        has_discount = np.random.random() < 0.25
        discount_percent = (
            np.random.choice([5, 10, 15, 20, 25, 30], p=[0.35, 0.25, 0.2, 0.1, 0.07, 0.03])
            if has_discount else 0
        )

        colors = ",".join(np.random.choice(colors_placeholder, size=np.random.randint(1, 3), replace=False))
        sizes = ",".join(np.random.choice(sizes_placeholder, size=np.random.randint(1, 3), replace=False))
        materials = ",".join(np.random.choice(materials_placeholder, size=np.random.randint(1, 3), replace=False))
        certifications = ",".join(np.random.choice(certifications_placeholder, size=np.random.randint(1, 2), replace=False))
        keywords = ",".join(np.random.choice(keywords_placeholder, size=np.random.randint(1, 3), replace=False))

        items.append([
            product_id,
            name,
            category,
            subcategory,
            price,
            stock,
            created_at.strftime("%Y-%m-%d"),
            is_new,
            brand,
            discount_percent,
            colors,
            sizes,
            materials,
            certifications,
            keywords,
        ])

    columns = [
        "ProductID",
        "Name",
        "Category",
        "Subcategory",
        "Price",
        "Stock",
        "CreatedAt",
        "IsNew",
        "Brand",
        "DiscountPercent",
        "Colors",
        "Sizes",
        "Materials",
        "Certifications",
        "Keywords",
    ]

    return pd.DataFrame(items, columns=columns), categories, brands


# -------------------------------
# PRODUCT RELATIONS (Point 6)
# -------------------------------
def generate_product_relations(items_df):
    rows = []

    by_sub = dict(tuple(items_df.groupby("Subcategory")))
    by_cat = dict(tuple(items_df.groupby("Category")))

    for _, row in tqdm(items_df.iterrows(), total=len(items_df), desc="Generating product relations"):
        pid = int(row["ProductID"])
        sub = row["Subcategory"]
        cat = row["Category"]

        subs_pool = [int(x) for x in by_sub.get(sub, pd.DataFrame())["ProductID"].tolist() if int(x) != pid]
        comp_pool = [
            int(x)
            for x in by_cat.get(cat, pd.DataFrame())["ProductID"].tolist()
            if int(x) != pid and items_df.loc[items_df["ProductID"] == int(x), "Subcategory"].iloc[0] != sub
        ]

        n_subs = min(len(subs_pool), np.random.randint(1, 4)) if subs_pool else 0
        n_comp = min(len(comp_pool), np.random.randint(1, 4)) if comp_pool else 0

        if n_subs > 0:
            for rid in np.random.choice(subs_pool, size=n_subs, replace=False):
                rows.append([pid, "substitute", int(rid), round(np.random.uniform(0.4, 0.95), 3)])

        if n_comp > 0:
            for rid in np.random.choice(comp_pool, size=n_comp, replace=False):
                rows.append([pid, "complement", int(rid), round(np.random.uniform(0.35, 0.9), 3)])

    return pd.DataFrame(rows, columns=["ProductID", "RelationType", "RelatedProductID", "Strength"])


# -------------------------------
# PRICE / PROMO HISTORY (Point 1)
# -------------------------------
def generate_price_history(items_df, weeks=20):
    rows = []
    today = datetime.utcnow().date()

    for _, row in tqdm(items_df.iterrows(), total=len(items_df), desc="Generating price history"):
        pid = int(row["ProductID"])
        base_price = float(row["Price"])

        curr = base_price
        for w in range(weeks, -1, -1):
            dt = today - timedelta(days=7 * w)
            drift = np.random.normal(0.0, 0.03)
            curr = max(1.0, curr * (1.0 + drift))

            promo = np.random.choice(PROMO_TYPES, p=[0.62, 0.08, 0.12, 0.08, 0.10])
            discount = 0
            if promo != "none":
                discount = int(np.random.choice([5, 10, 15, 20, 25, 30], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]))

            price_effective = round(curr * (1.0 - discount / 100.0), 2)
            rows.append([pid, dt.isoformat(), round(curr, 2), discount, promo, price_effective])

    return pd.DataFrame(rows, columns=["ProductID", "Date", "ListPrice", "DiscountPercent", "PromoType", "EffectivePrice"])


# -------------------------------
# SESSIONS GENERATION (Point 2)
# -------------------------------
def generate_sessions(clients_df, items_df, categories):
    sessions = []
    items_list = items_df["ProductID"].tolist()
    session_id = 1

    for _, client in tqdm(clients_df.iterrows(), total=len(clients_df), desc="Generating sessions"):
        preferred_cat = np.random.choice(categories)
        num_sessions = np.random.randint(1, MAX_SESSIONS + 1)
        for _ in range(num_sessions):
            session_date = datetime.utcnow() - timedelta(days=np.random.randint(0, 730))
            num_views = np.random.randint(1, MAX_VIEWS_PER_SESSION + 1)
            viewed_items = []

            intent_shift = np.random.random() < 0.18
            active_cat = np.random.choice(categories) if intent_shift else preferred_cat

            for _ in range(num_views):
                if np.random.random() < 0.72:
                    cat_items = items_df[items_df["Category"] == active_cat]["ProductID"].tolist()
                    viewed_items.append(int(np.random.choice(cat_items if cat_items else items_list)))
                else:
                    viewed_items.append(int(np.random.choice(items_list)))

            search_term = np.random.choice(SEARCH_TERMS)
            applied_filters = random.sample(["price", "brand", "rating", "discount", "stock"], k=np.random.randint(1, 4))
            sort_by = np.random.choice(SORT_OPTIONS)
            avg_dwell = round(float(np.random.uniform(8, 140)), 2)
            bounce = bool(np.random.random() < 0.12)

            sessions.append([
                session_id,
                int(client["ClientID"]),
                session_date.date().isoformat(),
                ",".join(map(str, viewed_items)),
                search_term,
                ",".join(applied_filters),
                sort_by,
                avg_dwell,
                bounce,
            ])
            session_id += 1

    # Anonymous sessions
    num_anon = int(len(sessions) * ANON_RATIO)
    for _ in range(num_anon):
        session_date = datetime.utcnow() - timedelta(days=np.random.randint(0, 730))
        num_views = np.random.randint(1, MAX_VIEWS_PER_SESSION + 1)
        viewed_items = np.random.choice(items_list, size=num_views, replace=False)
        search_term = np.random.choice(SEARCH_TERMS)
        applied_filters = random.sample(["price", "brand", "rating", "discount", "stock"], k=np.random.randint(1, 4))
        sort_by = np.random.choice(SORT_OPTIONS)
        avg_dwell = round(float(np.random.uniform(5, 120)), 2)
        bounce = bool(np.random.random() < 0.2)

        sessions.append([
            session_id,
            0,
            session_date.date().isoformat(),
            ",".join(map(str, viewed_items)),
            search_term,
            ",".join(applied_filters),
            sort_by,
            avg_dwell,
            bounce,
        ])
        session_id += 1

    cols = [
        "SessionID",
        "ClientID",
        "ViewDate",
        "ProductIDs",
        "SearchQuery",
        "AppliedFilters",
        "SortBy",
        "AvgDwellSec",
        "BounceFlag",
    ]
    return pd.DataFrame(sessions, columns=cols)


# -------------------------------
# WISHLIST SNAPSHOT GENERATION
# -------------------------------
def generate_wishlists(sessions_df):
    # NOTE: This dataset is now a saved-state snapshot for future realtime masks.
    # Training intentionally ignores wishlist/cart until the API-side realtime branch exists.
    wishlist_state = {}
    registered_sessions = sessions_df[sessions_df["ClientID"] > 0].copy()
    registered_sessions["ViewDate"] = pd.to_datetime(registered_sessions["ViewDate"], errors="coerce")
    registered_sessions = registered_sessions.sort_values(["ClientID", "ViewDate", "SessionID"])

    for _, row in tqdm(registered_sessions.iterrows(), total=len(registered_sessions), desc="Generating wishlist snapshots"):
        viewed_items = [int(pid) for pid in str(row["ProductIDs"]).split(",") if pid.strip().isdigit()]
        if not viewed_items:
            continue

        wishlist_size = max(1, int(len(viewed_items) * np.random.uniform(0.12, 0.35)))
        wishlist_items = random.sample(viewed_items, min(len(viewed_items), wishlist_size))
        saved_date = row["ViewDate"].date().isoformat() if pd.notna(row["ViewDate"]) else datetime.utcnow().date().isoformat()

        for item in wishlist_items:
            wishlist_state[(int(row["ClientID"]), int(item))] = [
                int(row["ClientID"]),
                int(item),
                saved_date,
            ]

    wishlist_rows = sorted(wishlist_state.values(), key=lambda r: (r[0], r[1]))
    return pd.DataFrame(wishlist_rows, columns=["ClientID", "ProductID", "SavedDate"])


# -------------------------------
# CART SNAPSHOT GENERATION
# -------------------------------
def generate_cart_state(sessions_df, items_df):
    price_map = dict(zip(items_df["ProductID"], items_df["Price"]))
    cart_state = {}
    registered_sessions = sessions_df[sessions_df["ClientID"] > 0].copy()
    registered_sessions["ViewDate"] = pd.to_datetime(registered_sessions["ViewDate"], errors="coerce")
    registered_sessions = registered_sessions.sort_values(["ClientID", "ViewDate", "SessionID"])

    for _, sess in tqdm(registered_sessions.iterrows(), total=len(registered_sessions), desc="Generating cart snapshots"):
        viewed_items = [int(pid) for pid in str(sess["ProductIDs"]).split(",") if pid.strip().isdigit()]
        if not viewed_items:
            continue

        cart_size = min(len(viewed_items), np.random.randint(0, 4))
        if cart_size == 0:
            continue

        selected_items = np.random.choice(viewed_items, size=cart_size, replace=False)
        snapshot_date = sess["ViewDate"].date().isoformat() if pd.notna(sess["ViewDate"]) else datetime.utcnow().date().isoformat()

        for pid in selected_items:
            qty = int(np.random.randint(1, 4))
            price = float(price_map.get(int(pid), 0.0))
            discount = int(np.random.choice([0, 5, 10, 15, 20], p=[0.5, 0.15, 0.15, 0.1, 0.1]))
            coupon_used = bool(np.random.random() < 0.25)
            cart_state[(int(sess["ClientID"]), int(pid))] = [
                int(sess["ClientID"]),
                int(pid),
                snapshot_date,
                qty,
                round(price, 2),
                discount,
                coupon_used,
            ]

    rows = sorted(cart_state.values(), key=lambda r: (r[0], r[1]))
    cols = ["ClientID", "ProductID", "SavedDate", "Quantity", "UnitPrice", "DiscountAtSnapshot", "CouponUsed"]
    return pd.DataFrame(rows, columns=cols)


# -------------------------------
# ORDERS GENERATION
# -------------------------------
def generate_orders(clients_df, sessions_df, items_df, relations_df):
    price_map = dict(zip(items_df["ProductID"], items_df["Price"]))

    comp_map = {}
    for _, rel in relations_df[relations_df["RelationType"] == "complement"].iterrows():
        comp_map.setdefault(int(rel["ProductID"]), []).append(int(rel["RelatedProductID"]))

    final_orders, final_items = [], []
    order_id_counter = 1

    client_ids = [cid for cid in clients_df["ClientID"] if cid in sessions_df["ClientID"].unique()]

    for cid in tqdm(client_ids, desc="Generating orders"):
        client_sessions = sessions_df[sessions_df["ClientID"] == cid]
        view_counter = Counter()
        for _, sess in client_sessions.iterrows():
            session_items = [int(pid) for pid in str(sess["ProductIDs"]).split(",") if pid.strip().isdigit()]
            for pid in set(session_items):
                view_counter[int(pid)] += 1

        num_orders = np.random.randint(0, 10)
        for _ in range(num_orders):
            if not view_counter:
                continue

            total_views = float(sum(view_counter.values()))
            weighted_pool = [
                (pid, 0.2 + (count / total_views))
                for pid, count in view_counter.items()
            ]
            if not weighted_pool:
                continue

            total_weight = sum(w for _, w in weighted_pool)
            normalized = [(pid, w / total_weight) for pid, w in weighted_pool]
            pids = [pid for pid, _ in normalized]
            probs = [w for _, w in normalized]

            num_products = np.random.randint(1, min(8, len(pids)) + 1)
            selected = list(np.random.choice(pids, size=num_products, replace=False, p=probs))

            # Complement boost to simulate bundles
            boosted = set(selected)
            for pid in selected:
                if pid in comp_map and np.random.random() < 0.35:
                    boosted.add(int(np.random.choice(comp_map[pid])))
            selected_products = list(boosted)

            order_date = datetime.utcnow().date() - timedelta(days=np.random.randint(0, 90))
            status = np.random.choice(["delivered", "cancelled", "returned"], p=[0.8, 0.1, 0.1])
            return_reason = np.random.choice(RETURN_REASONS) if status == "returned" else ""

            total_price = 0.0
            order_items = []
            for pid in selected_products:
                quantity = int(np.random.randint(1, 4))
                unit_price = float(price_map.get(pid, 0.0))
                total_price += unit_price * quantity

                rating = None
                review_text = ""
                if status != "cancelled" and np.random.random() < 0.72:
                    rating = int(np.random.choice([1, 2, 3, 4, 5], p=[0.06, 0.12, 0.26, 0.33, 0.23]))
                    review_text = np.random.choice(REVIEW_SNIPPETS)

                order_items.append([
                    order_id_counter,
                    int(pid),
                    quantity,
                    round(unit_price, 2),
                    rating,
                    review_text,
                ])

            final_orders.append([
                order_id_counter,
                int(cid),
                order_date.isoformat(),
                status,
                round(total_price, 2),
                return_reason,
            ])
            final_items.extend(order_items)
            order_id_counter += 1

    df_orders = pd.DataFrame(
        final_orders,
        columns=["OrderID", "ClientID", "OrderDate", "OrderStatus", "TotalPrice", "ReturnReason"],
    )
    df_items = pd.DataFrame(
        final_items,
        columns=["OrderID", "ProductID", "Quantity", "UnitPrice", "Rating", "ReviewText"],
    )
    return df_orders, df_items


# -------------------------------
# MAIN
# -------------------------------
def main():
    os.makedirs("../data", exist_ok=True)

    clients_df = generate_clients(NUM_CLIENTS)
    products_df, categories, _ = generate_items(NUM_ITEMS)

    relations_df = generate_product_relations(products_df)
    price_history_df = generate_price_history(products_df)


    sessions_df = generate_sessions(clients_df, products_df, categories)
    wishlist_df = generate_wishlists(sessions_df)
    cart_df = generate_cart_state(sessions_df, products_df)

    orders_df, order_items_df = generate_orders(
        clients_df,
        sessions_df,
        products_df,
        relations_df,
    )

    clients_df.to_csv("../data/clientes.csv", index=False)
    products_df.to_csv("../data/produtos.csv", index=False)
    sessions_df.to_csv("../data/viewed_products.csv", index=False)
    wishlist_df.to_csv("../data/wishlist.csv", index=False)
    orders_df.to_csv("../data/orders.csv", index=False)
    order_items_df.to_csv("../data/order_items.csv", index=False)

    # New generic scenario datasets
    relations_df.to_csv("../data/product_relations.csv", index=False)
    price_history_df.to_csv("../data/price_history.csv", index=False)
    # TODO(api-realtime): consume wishlist/cart snapshot files to build the realtime mask
    # once the API-side realtime branch is introduced.
    cart_df.to_csv("../data/cart_events.csv", index=False)

    print("Dataset generated with training-safe history signals plus wishlist/cart snapshots reserved for future realtime API masks.")


if __name__ == "__main__":
    main()
