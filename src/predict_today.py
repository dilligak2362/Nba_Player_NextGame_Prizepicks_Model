import pandas as pd
from pathlib import Path
from datetime import datetime

# ===========================
# PATHS
# ===========================
PP_PATH = Path("data/props/prizepicks.json")
UD_PATH = Path("data/props/underdog.json")
SLEEPER_PATH = Path("data/props/sleeper.json")

SINGLES_PATH = Path("data/processed/model_predictions.csv")
COMBO_PATH = Path("data/processed/combo_predictions.csv")
FANTASY_PATH = Path("data/processed/fantasy_predictions.csv")

OUT_MAIN = Path("data/processed/merged_props_predictions.csv")
OUT_SLEEPER = Path("data/processed/merged_sleeper_predictions.csv")

OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)


# ===========================
# LOAD BOOKS
# ===========================
def load_book(path, name):
    if not path.exists():
        print(f"⚠️ Missing {name} props file")
        return pd.DataFrame()
    df = pd.read_json(path)
    df["source"] = name
    return df


# ===========================
# LOAD MODEL PREDICTIONS (FRESH)
# ===========================
def load_predictions():
    frames = []

    for path in [SINGLES_PATH, COMBO_PATH, FANTASY_PATH]:
        if path.exists():
            print(f"📥 Loading predictions: {path} (modified {datetime.fromtimestamp(path.stat().st_mtime)})")
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("❌ No model prediction files found")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"player", "stat", "model_prediction", "proj_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"❌ Predictions missing columns: {missing}")

    df["player"] = df["player"].astype(str).str.strip()
    df["stat"] = df["stat"].astype(str).str.upper().str.strip()
    df["model_prediction"] = pd.to_numeric(df["model_prediction"], errors="coerce")
    df["proj_min"] = pd.to_numeric(df["proj_min"], errors="coerce")

    return df


# ===========================
# STAT NORMALIZER
# ===========================
def normalize_stat(val):
    val = str(val).upper().replace(" ", "").strip()

    basic = {
        "POINTS": "PTS",
        "PTS": "PTS",
        "REBOUNDS": "REB",
        "REB": "REB",
        "ASSISTS": "AST",
        "AST": "AST",
        "STEALS": "STL",
        "STL": "STL",
        "BLOCKS": "BLK",
        "BLK": "BLK",
        "TURNOVERS": "TO",
        "TURNOVER": "TO",
        "TO": "TO",
        "FANTASY": "FANTASY",
        "FANTASYPOINTS": "FANTASY",
    }

    if val in basic:
        return basic[val]

    if "+" in val:
        parts = set(
            val.replace("POINTS", "PTS")
               .replace("REBOUNDS", "REB")
               .replace("ASSISTS", "AST")
               .split("+")
        )

        if {"PTS", "REB", "AST"}.issubset(parts): return "PRA"
        if {"PTS", "REB"}.issubset(parts): return "PR"
        if {"PTS", "AST"}.issubset(parts): return "PA"
        if {"REB", "AST"}.issubset(parts): return "RA"

    return val


# ===========================
# MERGE ENGINE
# ===========================
def merge_book(props_df, preds_df, out_path):
    if props_df.empty:
        print(f"⚠️ No props for {out_path.name}")
        return

    props = props_df.copy()
    preds = preds_df.copy()

    props["player"] = props["player"].astype(str).str.strip()
    props["stat"] = props["stat"].apply(normalize_stat)

    if "line" in props.columns:
        props.rename(columns={"line": "book_line"}, inplace=True)

    props["book_line"] = pd.to_numeric(props["book_line"], errors="coerce")
    preds["stat"] = preds["stat"].apply(normalize_stat)

    preds = preds[["player", "stat", "model_prediction", "proj_min"]]

    merged = props.merge(preds, on=["player", "stat"], how="left")

    merged["edge"] = merged["model_prediction"] - merged["book_line"]

    merged["direction"] = merged.apply(
        lambda r: "OVER" if r["edge"] > 0 else "UNDER" if r["edge"] < 0 else "PUSH",
        axis=1
    )

    cols = [
        "player",
        "team",
        "stat",
        "book_line",
        "model_prediction",
        "proj_min",
        "edge",
        "direction",
        "source",
        "prop_type",
    ]

    cols = [c for c in cols if c in merged.columns]

    final = (
        merged
        .dropna(subset=["model_prediction", "book_line"])
        .loc[:, cols]
        .sort_values(by="edge", ascending=False)
        .reset_index(drop=True)
    )

    final.to_csv(out_path, index=False)

    print(f"\n💾 Saved merged board → {out_path}")
    print(f"Rows: {len(final)}")
    print("\nTop 10 edges:")
    print(final.head(10))


# ===========================
# MAIN
# ===========================
def merge_all():
    preds = load_predictions()

    pp = load_book(PP_PATH, "PrizePicks")
    ud = load_book(UD_PATH, "Underdog")
    sleeper = load_book(SLEEPER_PATH, "Sleeper")

    # Main board
    frames_main = []
    if not pp.empty:
        frames_main.append(pp)
    if not ud.empty:
        frames_main.append(ud)

    if frames_main:
        main_props = pd.concat(frames_main, ignore_index=True)
        merge_book(main_props, preds, OUT_MAIN)
    else:
        print("⚠️ No PrizePicks or Underdog props available")

    # Sleeper board
    if not sleeper.empty:
        merge_book(sleeper, preds, OUT_SLEEPER)
    else:
        print("⚠️ No Sleeper props available")


if __name__ == "__main__":
    merge_all()