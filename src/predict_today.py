# src/predict_today.py
import os
import time
import requests
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_URL = "https://api.balldontlie.io/v1"
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")
BALLDONTLIE_HEADERS = {"Authorization": BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}

DATASET_ADJ = Path("data/processed/inference_dataset_adjusted.csv")
DATASET_BASE = Path("data/processed/inference_dataset.csv")

MODELS_DIR = Path("models")

OUT_SINGLE = Path("data/processed/model_predictions.csv")
OUT_COMBO = Path("data/processed/combo_predictions.csv")
OUT_FANTASY = Path("data/processed/fantasy_predictions.csv")
OUT_SINGLE.parent.mkdir(parents=True, exist_ok=True)

def safe_get(endpoint: str, params=None, max_retries: int = 6):
    params = params or {}
    url = f"{BASE_URL}/{endpoint}"

    for i in range(max_retries):
        try:
            r = requests.get(url, headers=BALLDONTLIE_HEADERS, params=params, timeout=15)

            if r.status_code == 429:
                wait = 2 + i * 2
                print(f"⚠️ Rate limited (429). Retry {i+1}/{max_retries} in {wait}s…")
                time.sleep(wait)
                continue

            if r.status_code == 401:
                print("❌ 401 Unauthorized. BALLDONTLIE_API_KEY missing or invalid.")
                return None

            r.raise_for_status()
            return r.json()

        except Exception as e:
            wait = 2 + i
            print(f"Network/API issue: {e}. Retry {i+1}/{max_retries} in {wait}s…")
            time.sleep(wait)

    return None

def get_todays_games():
    today = datetime.now().strftime("%Y-%m-%d")

    if not BALLDONTLIE_API_KEY:
        print("⚠️ BALLDONTLIE_API_KEY not set. Using all teams in dataset.")
        return []

    games = []
    page = 1

    while True:
        data = safe_get("games", params={"dates[]": today, "per_page": 100, "page": page})
        if not data:
            return []

        games.extend(data.get("data", []))
        meta = data.get("meta", {}) or {}
        if page >= meta.get("total_pages", 1):
            break
        page += 1

    print(f"\n📅 Games today: {len(games)}")
    return games

def load_model(name: str):
    path = MODELS_DIR / f"{name}_rf.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return load(path)

def build_feature_vector(row, feature_names, override_min=None):
    x = []
    for c in feature_names:
        if c == "min" and override_min is not None:
            x.append(float(override_min))
        else:
            v = row.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                x.append(0.0)
            else:
                try:
                    x.append(float(v))
                except:
                    x.append(0.0)
    return np.array(x).reshape(1, -1)

def predict():
    print("\n🔮 Running prediction engine...\n")

    dataset_path = DATASET_ADJ if DATASET_ADJ.exists() else DATASET_BASE
    if not dataset_path.exists():
        raise FileNotFoundError("Missing inference dataset (base or adjusted). Run build_inference_dataset + injury adjustment first.")

    print(f"📄 Using inference dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "player_name" in df.columns and "player" not in df.columns:
        df.rename(columns={"player_name": "player"}, inplace=True)

    if "team" not in df.columns:
        raise ValueError("Inference dataset missing required column: team")

    games = get_todays_games()

    if not games:
        teams_today = set(df["team"].dropna().astype(str).str.upper().unique())
    else:
        teams_today = set()
        for g in games:
            if isinstance(g.get("home_team"), dict):
                teams_today.add(g["home_team"].get("abbreviation"))
            if isinstance(g.get("visitor_team"), dict):
                teams_today.add(g["visitor_team"].get("abbreviation"))

    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df = df[df["team"].isin(teams_today)].copy()

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.sort_values("date").groupby("player", as_index=False).tail(1)

    print(f"Players available for prediction: {len(df)}")

    stat_models = {
        "PTS": load_model("pts"),
        "REB": load_model("reb"),
        "AST": load_model("ast"),
        "STL": load_model("stl"),
        "BLK": load_model("blk"),
        "TO": load_model("to"),
    }
    minutes_model = load_model("minutes")

    singles_rows, combo_rows, fantasy_rows = [], [], []

    for _, row in df.iterrows():
        # Predict minutes
        min_features = list(minutes_model.feature_names_in_)
        x_min = build_feature_vector(row, min_features)
        proj_min = float(minutes_model.predict(x_min)[0])
        proj_min = max(10, min(40, proj_min))

        preds = {}

        # Stat predictions
        for stat, model in stat_models.items():
            x = build_feature_vector(row, model.feature_names_in_, override_min=proj_min)
            pred = float(model.predict(x)[0])
            pred = max(0.0, pred)

            # Smart TO fallback
            if stat == "TO" and pred < 0.25:
                base_to = None
                if "roll_avg_to" in row and row["roll_avg_to"] > 0:
                    base_to = row["roll_avg_to"]
                elif "to" in row and row["to"] > 0:
                    base_to = row["to"]

                if base_to is not None and "min" in row and row["min"] > 5:
                    to_per_min = base_to / row["min"]
                    pred = to_per_min * proj_min
                else:
                    pred = 0.095 * proj_min

                pred = max(0.5, min(7.5, pred))

            preds[stat] = pred

        # Superstar correction (keeps model from undercutting stars too hard)
        if "roll_avg_pts" in row and preds["PTS"] > 20:
            preds["PTS"] = max(preds["PTS"], float(row["roll_avg_pts"]) * 0.92)
        if "roll_avg_ast" in row and preds["AST"] > 5:
            preds["AST"] = max(preds["AST"], float(row["roll_avg_ast"]) * 0.92)
        if "roll_avg_reb" in row and preds["REB"] > 5:
            preds["REB"] = max(preds["REB"], float(row["roll_avg_reb"]) * 0.92)
        if "roll_avg_to" in row and preds["TO"] > 2:
            preds["TO"] = max(preds["TO"], float(row["roll_avg_to"]) * 0.92)

        # Save singles
        for stat, pred in preds.items():
            singles_rows.append({
                "player": row["player"],
                "team": row["team"],
                "stat": stat,
                "model_prediction": round(pred, 3),
                "proj_min": round(proj_min, 1),
            })

        # Combos
        combo_map = {
            "PR": preds["PTS"] + preds["REB"],
            "PA": preds["PTS"] + preds["AST"],
            "RA": preds["REB"] + preds["AST"],
            "PRA": preds["PTS"] + preds["REB"] + preds["AST"],
        }
        for k, v in combo_map.items():
            combo_rows.append({
                "player": row["player"],
                "team": row["team"],
                "stat": k,
                "model_prediction": round(float(v), 3),
                "proj_min": round(proj_min, 1),
            })

        # Fantasy
        fantasy = (
            preds["PTS"]
            + preds["REB"] * 1.2
            + preds["AST"] * 1.5
            + preds["STL"] * 3.0
            + preds["BLK"] * 3.0
        )
        fantasy_rows.append({
            "player": row["player"],
            "team": row["team"],
            "stat": "FANTASY",
            "model_prediction": round(float(fantasy), 3),
            "proj_min": round(proj_min, 1),
        })

    pd.DataFrame(singles_rows).to_csv(OUT_SINGLE, index=False)
    pd.DataFrame(combo_rows).to_csv(OUT_COMBO, index=False)
    pd.DataFrame(fantasy_rows).to_csv(OUT_FANTASY, index=False)

    print(f"\n✅ Saved singles → {OUT_SINGLE}")
    print(f"✅ Saved combos → {OUT_COMBO}")
    print(f"✅ Saved fantasy → {OUT_FANTASY}")

if __name__ == "__main__":
    predict()