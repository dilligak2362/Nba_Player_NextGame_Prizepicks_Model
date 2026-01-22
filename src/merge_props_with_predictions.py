# src/merge_props_with_predictions.py
import pandas as pd
from pathlib import Path

PP_PATH = Path("data/props/prizepicks.json")
UD_PATH = Path("data/props/underdog.json")

SINGLES_PATH = Path("data/processed/model_predictions.csv")
COMBO_PATH = Path("data/processed/combo_predictions.csv")
FANTASY_PATH = Path("data/processed/fantasy_predictions.csv")

OUT = Path("data/processed/merged_props_predictions.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_pp():
    if not PP_PATH.exists():
        print("❌ PrizePicks file missing")
        return pd.DataFrame()
    df = pd.read_json(PP_PATH)
    df["source"] = "PrizePicks"
    return df

def load_ud():
    if not UD_PATH.exists():
        return pd.DataFrame()
    df = pd.read_json(UD_PATH)
    df["source"] = "Underdog"
    return df

def load_predictions():
    frames = []
    if SINGLES_PATH.exists():
        frames.append(pd.read_csv(SINGLES_PATH))
    if COMBO_PATH.exists():
        frames.append(pd.read_csv(COMBO_PATH))
    if FANTASY_PATH.exists():
        frames.append(pd.read_csv(FANTASY_PATH))

    if not frames:
        raise FileNotFoundError("❌ No model prediction files found")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"player", "stat", "model_prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")

    df["player"] = df["player"].astype(str).str.strip()
    df["stat"] = df["stat"].astype(str).str.upper().str.strip()
    df["model_prediction"] = pd.to_numeric(df["model_prediction"], errors="coerce")

    # ensure these exist (important for downstream)
    if "team" not in df.columns:
        df["team"] = ""
    if "proj_min" not in df.columns:
        df["proj_min"] = 0.0

    return df

def normalize_stat(val):
    if val is None:
        return ""
    val = str(val).upper().replace(" ", "").strip()

    basic = {
        "POINTS": "PTS", "PTS": "PTS",
        "REBOUNDS": "REB", "REB": "REB",
        "ASSISTS": "AST", "AST": "AST",
        "STEALS": "STL", "STL": "STL",
        "BLOCKS": "BLK", "BLK": "BLK",
        "TURNOVERS": "TO", "TURNOVER": "TO", "TO": "TO",
        "FANTASY": "FANTASY", "FANTASYPOINTS": "FANTASY",
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

def merge_all():
    preds = load_predictions()
    pp, ud = load_pp(), load_ud()

    frames = []
    if not pp.empty: frames.append(pp)
    if not ud.empty: frames.append(ud)

    if not frames:
        print("❌ No sportsbook props available")
        return

    props = pd.concat(frames, ignore_index=True)

    props.columns = [c.lower().strip() for c in props.columns]
    props["player"] = props["player"].astype(str).str.strip()

    if "stat" not in props.columns:
        raise ValueError("Sportsbook props missing 'stat' column")

    props["stat"] = props["stat"].apply(normalize_stat)

    if "line" in props.columns and "book_line" not in props.columns:
        props.rename(columns={"line": "book_line"}, inplace=True)

    if "book_line" not in props.columns:
        raise ValueError("Sportsbook props missing book line column (expected 'line' or 'book_line')")

    props["book_line"] = pd.to_numeric(props["book_line"], errors="coerce")

    preds["stat"] = preds["stat"].apply(normalize_stat)

    merged = props.merge(preds, on=["player", "stat"], how="left", suffixes=("", "_pred"))

    # If prop team missing, fill from prediction team
    if "team" not in merged.columns:
        merged["team"] = merged.get("team_pred", "")
    else:
        merged["team"] = merged["team"].fillna(merged.get("team_pred", ""))

    # Keep proj_min from predictions
    if "proj_min" not in merged.columns:
        merged["proj_min"] = merged.get("proj_min_pred", 0.0)
    merged["proj_min"] = pd.to_numeric(merged["proj_min"], errors="coerce").fillna(0.0)

    merged["edge"] = merged["model_prediction"] - merged["book_line"]

    merged["direction"] = merged.apply(
        lambda r: "OVER" if r["model_prediction"] > r["book_line"]
        else "UNDER" if r["model_prediction"] < r["book_line"]
        else "PUSH",
        axis=1
    )

    # Identify prop_type if not present
    if "prop_type" not in merged.columns:
        merged["prop_type"] = "standard"
        merged.loc[merged["stat"].isin(["PR", "PA", "RA", "PRA"]), "prop_type"] = "combo"
        merged.loc[merged["stat"].isin(["FANTASY"]), "prop_type"] = "fantasy"

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
        .rename(columns={"stat": "prop"})
        .sort_values(by="edge", ascending=False)
        .reset_index(drop=True)
    )

    final.to_csv(OUT, index=False)

    print(f"\n💾 Saved merged props → {OUT}")
    print(f"Rows: {len(final)}")
    print("\nTop 10 edges:")
    print(final.head(10).to_string(index=False))

if __name__ == "__main__":
    merge_all()