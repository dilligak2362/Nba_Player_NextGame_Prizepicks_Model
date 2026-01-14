import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/historical_boxscores.csv")
OUT = Path("data/processed/minutes_training_dataset.csv")

SEASON = 2024
ROLL_WINDOW = 5


# ============================
# LOAD RAW DATA
# ============================
def load_data():
    print("Loading raw data...")
    df = pd.read_csv(RAW)
    df.columns = [c.lower() for c in df.columns]

    needed = [
        "season", "player_id", "player_name",
        "team", "opponent", "date",
        "pts", "reb", "ast", "stl", "blk", "to",
        "fga", "fta", "oreb", "min"
    ]

    for col in needed:
        if col not in df.columns:
            raise Exception(f"Missing {col}")

    df = df[df["season"] == SEASON].copy()
    df["date"] = pd.to_datetime(df["date"])
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
# FEATURE ENGINEERING
# ============================
def build_features(df):
    out = []

    for pid, gp in df.groupby("player_id"):
        gp = gp.sort_values("date").copy()

        safe_min = gp["min"].replace(0, 0.1)

        # Rolling minutes
        gp["roll_min"] = gp["min"].rolling(ROLL_WINDOW, 1).mean()
        gp["lag1_min"] = gp["min"].shift(1)
        gp["lag2_min"] = gp["min"].shift(2)
        gp["ewm_min"] = gp["min"].ewm(span=5, adjust=False).mean()

        # Usage proxy
        gp["usage_proxy"] = (gp["fga"] + 0.44 * gp["fta"] + gp["to"]) / safe_min
        gp["roll_usage"] = gp["usage_proxy"].rolling(ROLL_WINDOW, 1).mean()

        # Role classification (rotation-aware)
        gp["role"] = np.select(
            [
                gp["roll_usage"] > 1.15,
                gp["roll_usage"] > 0.85,
                gp["roll_usage"] > 0.55,
            ],
            ["star", "starter", "rotation"],
            default="bench"
        )

        gp["is_star"] = (gp["role"] == "star").astype(int)
        gp["is_starter"] = (gp["role"] == "starter").astype(int)
        gp["is_rotation"] = (gp["role"] == "rotation").astype(int)

        # Games played
        gp["games_played"] = gp.groupby("player_id").cumcount()

        # Target = next game minutes
        gp["target_min"] = gp["min"].shift(-1)

        out.append(gp)

    return pd.concat(out)


# ============================
# FINALIZE
# ============================
def finalize(df):
    keep = [
        "player_id", "player_name", "team", "opponent", "date",
        "min",
        "roll_min", "lag1_min", "lag2_min", "ewm_min",
        "usage_proxy", "roll_usage",
        "is_star", "is_starter", "is_rotation",
        "games_played",
        "target_min"
    ]

    df = df[keep].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved minutes training dataset → {OUT}")
    print("Final shape:", df.shape)


def main():
    df = load_data()
    df = build_features(df)
    finalize(df)


if __name__ == "__main__":
    main()
