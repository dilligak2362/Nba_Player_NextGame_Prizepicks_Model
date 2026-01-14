import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/historical_boxscores.csv")
OUT = Path("data/processed/inference_dataset.csv")

SEASON = 2025
ROLL_WINDOW = 5


# ============================
# LOAD DATA
# ============================
def load_data():
    print("Loading raw data...")
    df = pd.read_csv(RAW)
    df.columns = [c.lower() for c in df.columns]

    needed = [
        "season", "player_id", "player_name",
        "team", "opponent", "date",
        "pts", "reb", "ast", "stl", "blk", "to",
        "fga", "fta", "oreb",
        "min"
    ]

    for col in needed:
        if col not in df.columns:
            raise Exception(f"❌ Missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["season"] == SEASON].copy()
    df = df.sort_values(["player_id", "date"])

    def parse_minutes(m):
        if isinstance(m, str) and ":" in m:
            mm, ss = m.split(":")
            return float(mm) + float(ss) / 60
        return float(m)

    df["min"] = df["min"].apply(parse_minutes)
    df = df[df["min"] > 0].copy()

    return df


# ============================
# OPPONENT CONTEXT + PACE
# ============================
def add_opponent_context(df):
    print("Adding opponent defensive context + pace proxy...")

    df["poss"] = df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["to"]

    opp = (
        df.groupby("opponent")[["pts", "reb", "ast", "stl", "blk", "to", "poss"]]
        .mean()
        .reset_index()
        .rename(columns={
            "pts": "opp_allow_pts",
            "reb": "opp_allow_reb",
            "ast": "opp_allow_ast",
            "stl": "opp_allow_stl",
            "blk": "opp_allow_blk",
            "to": "opp_allow_to",
            "poss": "opp_pace_proxy"
        })
    )

    df = df.merge(opp, on="opponent", how="left")
    return df


# ============================
# FEATURE ENGINEERING
# ============================
def build_features(df):
    print("Building inference feature set...")

    out = []

    for pid, gp in df.groupby("player_id"):
        gp = gp.sort_values("date").copy()
        safe_min = gp["min"].replace(0, 0.1)

        # ----------------------------
        # Rolling box stats
        # ----------------------------
        gp["roll_avg_pts"] = gp["pts"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_reb"] = gp["reb"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_ast"] = gp["ast"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_stl"] = gp["stl"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_blk"] = gp["blk"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_to"]  = gp["to"].rolling(ROLL_WINDOW, 1).mean()

        # ----------------------------
        # Lag box stats
        # ----------------------------
        gp["lag1_pts"] = gp["pts"].shift(1)
        gp["lag1_reb"] = gp["reb"].shift(1)
        gp["lag1_ast"] = gp["ast"].shift(1)
        gp["lag1_stl"] = gp["stl"].shift(1)
        gp["lag1_blk"] = gp["blk"].shift(1)
        gp["lag1_to"]  = gp["to"].shift(1)

        # ----------------------------
        # Minutes model features
        # ----------------------------
        gp["roll_min"] = gp["min"].rolling(ROLL_WINDOW, 1).mean()
        gp["lag1_min"] = gp["min"].shift(1)
        gp["lag2_min"] = gp["min"].shift(2)
        gp["ewm_min"] = gp["min"].ewm(span=5, adjust=False).mean()

        # ----------------------------
        # Usage model features
        # ----------------------------
        gp["usage_proxy"] = (gp["fga"] + 0.44 * gp["fta"] + gp["to"]) / safe_min
        gp["lag1_usage"] = gp["usage_proxy"].shift(1)
        gp["roll_usage"] = gp["usage_proxy"].rolling(ROLL_WINDOW, 1).mean()
        gp["ewm_usage"] = gp["usage_proxy"].ewm(span=5, adjust=False).mean()

        # ----------------------------
        # Games played
        # ----------------------------
        gp["games_played"] = gp.groupby("player_id").cumcount()

        out.append(gp)

    df2 = pd.concat(out, ignore_index=True)

    # Last game per player for inference
    df2 = (
        df2.sort_values("date")
           .groupby("player_id", as_index=False)
           .tail(1)
    )

    return df2


# ============================
# FINALIZE
# ============================
def finalize(df):
    df = df.rename(columns={"player_name": "player"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved inference dataset → {OUT}")
    print("Final shape:", df.shape)


# ============================
# MAIN
# ============================
def main():
    df = load_data()
    df = add_opponent_context(df)
    df = build_features(df)
    finalize(df)


if __name__ == "__main__":
    main()