# FastAPI Restaurant Recommender API
# Hybrid: Content + Basket + SVD (On-the-fly similarity, serving subset, menu-only option)

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pickle
import pandas as pd
from difflib import get_close_matches
from pathlib import Path
import json

# ----------------------------
# Paths (match training script)
# ----------------------------
# This file lives at: <repo_root>/recommender/app/main.py
# So BASE_DIR should be: <repo_root>/recommender
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent            # /.../recommender
RAW_DIR = BASE_DIR / "data"
PROCESSED_DIR = RAW_DIR / "processed"

# Optional: debug print once at startup
print(f"📁 BASE_DIR = {BASE_DIR}")
print(f"📁 PROCESSED_DIR = {PROCESSED_DIR}")

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load Artifacts (serving subset)
# ----------------------------
try:
    recipes = pd.read_pickle(PROCESSED_DIR / "recipes.pkl")  # serving subset
    with open(PROCESSED_DIR / "tfidf_norm.pkl", "rb") as f:
        tfidf_norm = pickle.load(f)
    with open(PROCESSED_DIR / "ingredient_index.pkl", "rb") as f:
        ingredient_index = pickle.load(f)
    with open(PROCESSED_DIR / "svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
    with open(PROCESSED_DIR / "name_map.pkl", "rb") as f:
        name_map = pickle.load(f)
    with open(PROCESSED_DIR / "user_to_seen.pkl", "rb") as f:
        user_to_seen = pickle.load(f)
    with open(PROCESSED_DIR / "my_menu.json", "r") as f:
        MY_MENU = json.load(f)   # { recipe_id_str: menu_name }
    MY_MENU_IDS = list(map(int, MY_MENU.keys()))
except FileNotFoundError as e:
    raise RuntimeError(
        f"❌ Model files not found. Run hybrid_train.py first.\nMissing file: {e}"
    )

recipes["name"] = recipes["name"].astype(str)
recipes["ingredients"] = recipes["ingredients"].astype(str)

# Precompute menu-only DataFrame (subset of recipes)
menu_df = recipes[recipes["id"].isin(MY_MENU_IDS)].copy()
menu_df.reset_index(drop=True, inplace=True)

# ----------------------------
# Category-level complement rules (aligned with Mongo menu categories)
# ----------------------------
CATEGORY_COMPLEMENT_MAP = {
    "main course": ["Indian Breads", "Rice", "Fried Rice", "Extras", "Beverage"],
    "snacks": ["Beverage", "Mojito", "Krushers"],
    "momos": ["Beverage", "Mojito", "Snacks", "Fried Rice", "Noodles"],
    "indian veg thali": ["Beverage", "Extras"],
    "rice bowl": ["Beverage", "Snacks"],
    "indian breads": ["Main Course"],
    "krushers": ["Snacks", "Momos"],
    "mojito": ["Snacks", "Momos"],
    "beverage_lassi": ["Main Course", "Indian Veg Thali"],
    "breakfast": ["Beverage", "Extras"],
    "maggi": ["Snacks", "Beverage"],
    "south indian": ["Beverage", "Extras"],
    "burger & fries": ["Beverage", "Snacks", "Krushers"],
    "noodles": ["Fried Rice", "Snacks", "Beverage"],
    "wraps": ["Beverage", "Snacks"],
    "fried rice": ["Noodles", "Snacks", "Beverage"],
    "pasta": ["Beverage", "Snacks"],
    "sandwich": ["Beverage", "Krushers"],
    "pizza": ["Beverage", "Krushers"],
    "chinese combo": ["Beverage", "Snacks"],
}

# ----------------------------
# FastAPI App + CORS
# ----------------------------
app = FastAPI(title="Restaurant Recommender API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ----------------------------
# Utility
# ----------------------------
def get_food_image(name: str) -> str:
    query = f"food {name.lower().replace(' ', '-')}"
    return f"https://source.unsplash.com/300x300/?{query}"

def get_basket_neighbors(idx: int, topk: int = 20):
    target = set(recipes.at[idx, "ingredients"].split())
    scores = {}
    for ing in target:
        for r in ingredient_index.get(ing, []):
            if r != idx:
                scores[r] = scores.get(r, 0) + 1
    if not scores:
        return [], []
    items = sorted(scores.items(), key=lambda x: -x[1])[:topk]
    b_idx = [i for i, _ in items]
    b_scores = [s for _, s in items]
    return b_idx, b_scores

# ----------------------------
# Request Schema
# ----------------------------
class RecommendQuery(BaseModel):
    query: str
    user_id: Optional[str] = None
    topk: int = 10

# ----------------------------
# Core hybrid recommender (serving subset, capped SVD)
# ----------------------------
def hybrid_recommend_core(
    recipe_name: str,
    topk: int = 10,
    user_id: Optional[str] = None,
    svd_cap: int = 200,
):
    if not recipe_name.strip():
        return pd.DataFrame(columns=["id", "name"])

    match = get_close_matches(recipe_name.lower(), name_map.keys(), n=1, cutoff=0.5)
    if not match:
        return pd.DataFrame(columns=["id", "name"])
    idx = name_map[match[0]]

    sims = tfidf_norm[idx] @ tfidf_norm.T
    sims[0, idx] = 0
    c_scores = sims.toarray().ravel()
    if c_scores.max() > 0:
        c_scores = c_scores / c_scores.max()
    c_idx = c_scores.argsort()[::-1][:50]
    c_map = {int(i): float(c_scores[i]) for i in c_idx}

    b_idx, b_scores = get_basket_neighbors(idx)
    if len(b_scores) > 0:
        max_b = max(b_scores)
        b_scores = [s / max_b for s in b_scores]
    b_map = {int(i): float(s) for i, s in zip(b_idx, b_scores)}

    candidates = set(c_idx) | set(b_idx)
    if idx in candidates:
        candidates.remove(idx)

    scores = {}
    for i in candidates:
        scores[i] = 0.4 * c_map.get(i, 0.0) + 0.3 * b_map.get(i, 0.0)

    if user_id is not None and user_id != "":
        top_cands = sorted(scores.items(), key=lambda x: -x[1])[:svd_cap]
        top_ids = [i for i, _ in top_cands]
        cand_scores = {i: scores[i] for i in top_ids}

        seen = user_to_seen.get(str(user_id), set())
        for i in top_ids:
            rid = int(recipes.at[i, "id"])
            if rid not in seen:
                try:
                    cand_scores[i] += 0.3 * svd_model.predict(str(user_id), rid).est
                except Exception:
                    pass

        scores = cand_scores

    if not scores:
        return pd.DataFrame(columns=["id", "name"])

    top_items = sorted(scores.items(), key=lambda x: -x[1])[:topk]
    idxs = [i for i, _ in top_items]
    return recipes.loc[idxs, ["id", "name"]]

# ----------------------------
# Helper: infer menu category from canonical name
# ----------------------------
def infer_category_from_name(name: str) -> str:
    low = name.lower()

    if "thali" in low:
        return "Indian Veg Thali"
    if "rice bowl" in low:
        return "Rice Bowl"
    if any(k in low for k in [
        "dal fry", "aloo jeera", "dum aloo",
        "dal makhni", "chana masala",
        "matar paneer", "mushroom masala",
        "shahi paneer", "paneer lababdar"
    ]):
        return "Main Course"
    if any(k in low for k in ["biryani", "jeera rice", "pulao"]):
        return "Rice"
    if "fried rice" in low or "singapore rice" in low:
        return "Fried Rice"
    if "momo" in low or "momos" in low:
        return "Momos"
    if any(k in low for k in ["roti", "prantha", "parantha"]):
        return "Indian Breads"
    if any(k in low for k in ["curd", "raita", "papad", "salad"]):
        return "Extras"
    if "mojito" in low:
        return "Mojito"
    if "krusher" in low:
        return "Krushers"
    if any(k in low for k in ["tea", "lassi", "coffee", "shake", "soft drinks", "soft drink"]):
        return "Beverage"

    if any(k in low for k in ["maggi"]):
        return "Maggi"
    if any(k in low for k in ["dosa", "uttapam", "idli", "sambhar", "punugullu"]):
        return "South Indian"
    if any(k in low for k in ["burger", "fries"]):
        return "Burger & Fries"
    if any(k in low for k in ["noodles", "chowmein"]):
        return "Noodles"
    if "wrap" in low:
        return "Wraps"
    if "pasta" in low:
        return "Pasta"
    if "sandwich" in low:
        return "Sandwich"
    if "pizza" in low:
        return "Pizza"
    if any(k in low for k in ["breakfast", "poha", "upma", "bhatura"]):
        return "Breakfast"
    if any(k in low for k in ["chinese platter", "chinese combo"]):
        return "Chinese Combo"
    if any(k in low for k in ["corn", "spring roll", "manchurian", "chilli", "garlic bread",
                              "paneer 65", "cheese finger", "champ"]):
        return "Snacks"

    return ""

# ----------------------------
# Menu-only recommender (scores only menu_df + category complements)
# ----------------------------
def recommend_from_menu(
    query: str,
    topk: int = 10,
    user_id: Optional[str] = None,
):
    if not query.strip():
        return []

    match = get_close_matches(query.lower(), name_map.keys(), n=1, cutoff=0.3)
    if not match:
        return []

    anchor_idx = name_map[match[0]]

    sims = tfidf_norm[anchor_idx] @ tfidf_norm.T
    sims[0, anchor_idx] = 0
    c_scores = sims.toarray().ravel()
    if c_scores.max() > 0:
        c_scores = c_scores / c_scores.max()
    c_map = {i: float(c_scores[i]) for i in range(len(c_scores))}

    b_idx, b_scores = get_basket_neighbors(anchor_idx)
    if len(b_scores) > 0:
        max_b = max(b_scores)
        if max_b > 0:
            b_scores = [s / max_b for s in b_scores]
    b_map = {int(i): float(s) for i, s in zip(b_idx, b_scores)}

    menu_scores = {}
    for _, row in menu_df.iterrows():
        rid = int(row["id"])
        idx_list = recipes.index[recipes["id"] == rid].tolist()
        if not idx_list:
            continue
        idx = idx_list[0]
        base = 0.4 * c_map.get(idx, 0.0) + 0.3 * b_map.get(idx, 0.0)
        menu_scores[idx] = base

    anchor_menu_name = recipes.at[anchor_idx, "name"].strip()
    for rid_str, menu_name in MY_MENU.items():
        rid = int(rid_str)
        idx_list = recipes.index[recipes["id"] == rid].tolist()
        if not idx_list:
            continue
        idx = idx_list[0]
        if idx == anchor_idx:
            anchor_menu_name = menu_name
            break

    anchor_cat = infer_category_from_name(anchor_menu_name)
    anchor_cat_key = anchor_cat.strip().lower()

    if anchor_menu_name.strip().lower() == "lassi":
        rule_key = "beverage_lassi"
    else:
        rule_key = anchor_cat_key

    target_categories = CATEGORY_COMPLEMENT_MAP.get(rule_key, [])
    target_cat_keys = [c.lower() for c in target_categories]

    for _, row in menu_df.iterrows():
        rid = int(row["id"])
        idx_list = recipes.index[recipes["id"] == rid].tolist()
        if not idx_list:
            continue
        idx = idx_list[0]

        if idx == anchor_idx:
            continue

        menu_name = MY_MENU.get(str(rid), recipes.at[idx, "name"])
        item_cat = infer_category_from_name(menu_name)
        item_cat_key = item_cat.strip().lower()

        if item_cat_key in target_cat_keys:
            menu_scores[idx] = max(menu_scores.get(idx, 0.0), 0.8)

    if user_id is not None and user_id != "":
        seen = user_to_seen.get(str(user_id), set())
        for idx in list(menu_scores.keys()):
            rid = int(recipes.at[idx, "id"])
            if rid not in seen:
                try:
                    menu_scores[idx] += 0.3 * svd_model.predict(str(user_id), rid).est
                except Exception:
                    pass

    if not menu_scores:
        return []

    top_items = sorted(menu_scores.items(), key=lambda x: -x[1])[:topk]
    idxs = [i for i, _ in top_items]
    out = recipes.loc[idxs, ["id", "name"]].copy()

    def _canonical_name(rid: int, fallback_name: str) -> str:
        return MY_MENU.get(str(rid), fallback_name)

    out["name"] = out.apply(
        lambda row: _canonical_name(int(row["id"]), str(row["name"])),
        axis=1,
    )
    out["image_url"] = out["name"].apply(get_food_image)
    return out[["id", "name", "image_url"]].to_dict(orient="records")

# ----------------------------
# /recommend Endpoint (menu-only)
# ----------------------------
@app.post("/recommend")
def recommend(query: RecommendQuery):
    res = recommend_from_menu(
        query=query.query,
        topk=query.topk,
        user_id=query.user_id,
    )
    if not res:
        raise HTTPException(status_code=404, detail="No menu recommendations found")
    return res

# ----------------------------
# /recommend-menu Endpoint (explicit menu-only)
# ----------------------------
@app.post("/recommend-menu")
def recommend_menu(query: RecommendQuery):
    res = recommend_from_menu(
        query=query.query,
        topk=query.topk,
        user_id=query.user_id,
    )
    if not res:
        raise HTTPException(status_code=404, detail="No menu recommendations found")
    return res

# ----------------------------
# /suggest Endpoint
# ----------------------------
@app.get("/suggest", response_model=List[str])
def suggest(q: str = Query(..., min_length=1)):
    name_map_full = {n.lower(): n for n in recipes["name"]}
    matches = get_close_matches(q.lower(), name_map_full.keys(), n=10, cutoff=0.3)
    return [name_map_full[m] for m in matches]

# ----------------------------
# /top Endpoint
# ----------------------------
@app.get("/top", response_model=List[dict])
def top_recipes(topk: int = 10):
    top = menu_df.head(topk)[["id", "name"]].copy()

    def _canonical_name(rid: int, fallback_name: str) -> str:
        return MY_MENU.get(str(rid), fallback_name)

    top["name"] = top.apply(
        lambda row: _canonical_name(int(row["id"]), str(row["name"])),
        axis=1,
    )
    top["image_url"] = top["name"].apply(get_food_image)
    return top.to_dict(orient="records")

# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "✔ Restaurant Recommender API is running (menu-only)."}
