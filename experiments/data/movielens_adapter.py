#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np


GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


# Bump when the conversion changes so cached CSVs are rebuilt.
# v2: view sessions per (user, day) with true dates (enables temporal holdout).
ADAPTER_VERSION = "2"


def ensure_movielens_data(target_dir: str = os.path.join("data", "movielens")) -> str:
    """
    Downloads MovieLens-100k (if not already cached) and transforms it into the
    exact schema expected by the EARS hybrid preprocessing pipeline.
    """
    os.makedirs(target_dir, exist_ok=True)
    clientes_path = os.path.join(target_dir, "clientes.csv")
    produtos_path = os.path.join(target_dir, "produtos.csv")
    orders_path = os.path.join(target_dir, "orders.csv")
    order_items_path = os.path.join(target_dir, "order_items.csv")
    views_path = os.path.join(target_dir, "viewed_products.csv")
    version_path = os.path.join(target_dir, ".adapter_version")

    def _cached_version():
        if not os.path.exists(version_path):
            return None
        with open(version_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    if (
        all(os.path.exists(p) for p in [clientes_path, produtos_path, orders_path, order_items_path, views_path])
        and _cached_version() == ADAPTER_VERSION
    ):
        print(f"MovieLens-100k data already prepared at: {target_dir}")
        return target_dir

    print("Downloading MovieLens-100k from GroupLens...")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))

    # Read raw files
    data_bytes = z.read("ml-100k/u.data")
    ratings_df = pd.read_csv(
        io.BytesIO(data_bytes),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )

    user_bytes = z.read("ml-100k/u.user")
    users_df = pd.read_csv(
        io.BytesIO(user_bytes),
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        engine="python"
    )

    item_bytes = z.read("ml-100k/u.item")
    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    items_df = pd.read_csv(
        io.BytesIO(item_bytes),
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="python"
    )

    print(f"Loaded MovieLens: {len(users_df)} users, {len(items_df)} movies, {len(ratings_df)} ratings.")

    # 1. Generate clientes.csv
    # Schema: ClientID,Name,Email,Age,Gender,Location
    clientes_df = pd.DataFrame({
        "ClientID": users_df["user_id"],
        "Name": users_df["user_id"].apply(lambda uid: f"User_{uid}"),
        "Email": users_df["user_id"].apply(lambda uid: f"user{uid}@movielens.org"),
        "Age": users_df["age"],
        "Gender": users_df["gender"],
        "Location": users_df["occupation"].apply(lambda occ: f"Occ_{occ}")
    })
    clientes_df.to_csv(clientes_path, index=False)

    # 2. Generate produtos.csv
    # Derive dominant category and subcategory from genres
    def get_categories(row):
        active = [g for g in GENRES if row.get(g, 0) == 1]
        cat = active[0] if active else "General"
        subcat = active[1] if len(active) > 1 else cat
        return cat, subcat, ",".join(active)

    cat_tuples = [get_categories(row) for _, row in items_df.iterrows()]
    categories = [t[0] for t in cat_tuples]
    subcategories = [t[1] for t in cat_tuples]
    keywords = [t[2] for t in cat_tuples]

    produtos_df = pd.DataFrame({
        "ProductID": items_df["item_id"],
        "Name": items_df["title"],
        "Category": categories,
        "Subcategory": subcategories,
        "Price": 12.99,
        "Stock": 50,
        "CreatedAt": "2024-01-01",
        "IsNew": False,
        "Brand": [f"Studio_{c}" for c in categories],
        "DiscountPercent": 0,
        "Colors": "Standard",
        "Sizes": "Standard",
        "Materials": "Digital",
        "Certifications": "Universal",
        "Keywords": keywords
    })
    produtos_df.to_csv(produtos_path, index=False)

    # 3. Generate orders.csv and order_items.csv
    # Treat ratings >= 4 as positive purchases (Orders)
    positive_ratings = ratings_df[ratings_df["rating"] >= 4].copy()
    positive_ratings["OrderID"] = np.arange(1, len(positive_ratings) + 1)
    positive_ratings["OrderDate"] = pd.to_datetime(positive_ratings["timestamp"], unit="s").dt.strftime("%Y-%m-%d")

    orders_df = pd.DataFrame({
        "OrderID": positive_ratings["OrderID"],
        "ClientID": positive_ratings["user_id"],
        "OrderDate": positive_ratings["OrderDate"],
        "OrderStatus": "delivered",
        "TotalPrice": 12.99,
        "ReturnReason": ""
    })
    orders_df.to_csv(orders_path, index=False)

    order_items_df = pd.DataFrame({
        "OrderID": positive_ratings["OrderID"],
        "ProductID": positive_ratings["item_id"],
        "Quantity": 1,
        "UnitPrice": 12.99,
        "Rating": positive_ratings["rating"],
        "ReviewText": "Positive feedback"
    })
    order_items_df.to_csv(order_items_path, index=False)

    # 4. Generate viewed_products.csv
    # Every rating (any value) is an exposure. One session per (user, day), dated by the
    # rating timestamp, so that a temporal cutoff can exclude future interactions.
    views_list = []
    session_id = 1
    ratings_df = ratings_df.sort_values(["user_id", "timestamp"])
    ratings_df["day"] = pd.to_datetime(ratings_df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    for (uid, day), group in ratings_df.groupby(["user_id", "day"], sort=True):
        views_list.append({
            "SessionID": session_id,
            "ClientID": uid,
            "ViewDate": day,
            "ProductIDs": ",".join(str(pid) for pid in group["item_id"]),
            "SearchQuery": "movie search",
            "AppliedFilters": "rating",
            "SortBy": "popular",
            "AvgDwellSec": 45.0,
            "BounceFlag": False
        })
        session_id += 1

    views_df = pd.DataFrame(views_list)
    views_df.to_csv(views_path, index=False)

    # 5. Empty/minimal auxiliary files
    pd.DataFrame(columns=["ClientID", "ProductID", "AddedDate"]).to_csv(os.path.join(target_dir, "wishlist.csv"), index=False)
    pd.DataFrame(columns=["ClientID", "ProductID", "EventDate"]).to_csv(os.path.join(target_dir, "cart_events.csv"), index=False)
    pd.DataFrame(columns=["ProductID1", "ProductID2", "RelationType"]).to_csv(os.path.join(target_dir, "product_relations.csv"), index=False)
    pd.DataFrame(columns=["ProductID", "Price", "Date"]).to_csv(os.path.join(target_dir, "price_history.csv"), index=False)

    with open(version_path, "w", encoding="utf-8") as f:
        f.write(ADAPTER_VERSION)

    print(f"Successfully adapted MovieLens-100k into {target_dir} ({len(clientes_df)} clients, {len(produtos_df)} items, {len(orders_df)} orders).")
    return target_dir


if __name__ == "__main__":
    ensure_movielens_data()
