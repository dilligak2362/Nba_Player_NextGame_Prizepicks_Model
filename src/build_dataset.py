import pandas as pd
from pathlib import Path

RAW = Path("data/raw/historical_boxscores.csv")
OUT = Path("data/processed/training_dataset.csv")

TARGET_SEASON = 2024
ROLL_WINDOW = 5
MINUTES_CUTOFF = 10


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
        "fga", "fta", "oreb",
        "min"
    ]

    for col in needed:
        if col not in df.columns:
            raise Exception(f"❌ Missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    for col in ["pts", "reb", "ast", "stl", "blk", "to", "fga", "fta", "oreb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    def parse_minutes(m):
        if isinstance(m, str) and ":" in m:
            mm, ss = m.split(":")
            return float(mm) + float(ss) / 60
        return float(m)

    df["min"] = df["min"].apply(parse_minutes)
    df["is_home"] = 1

    df = df[df["season"] == TARGET_SEASON].copy()
    print(f"Rows after season filter: {len(df)}")

    return df


# ============================
# OPPONENT DEFENSE + PACE
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
# PLAYER IDENTITY FILTER
# ============================
def get_identity(df):
    print("Building player identity table...")

    id_df = (
        df.groupby(["player_id", "player_name"])
        .agg(
            team=("team", "last"),
            games_played=("min", "count"),
            avg_min=("min", "mean"),
        )
        .reset_index()
    )

    id_df = id_df[id_df["avg_min"] >= MINUTES_CUTOFF]
    id_df.rename(columns={"player_name": "player"}, inplace=True)

    print(f"Players kept: {len(id_df)}")
    return id_df


# ============================
# FEATURE ENGINEERING
# ============================
def build_features(df):
    feats = []

    for pid, gp in df.groupby("player_id"):
        gp = gp.sort_values("date").copy()

        safe_min = gp["min"].replace(0, 0.1)

        # Rolling box stats
        gp["roll_avg_pts"] = gp["pts"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_reb"] = gp["reb"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_ast"] = gp["ast"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_stl"] = gp["stl"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_blk"] = gp["blk"].rolling(ROLL_WINDOW, 1).mean()
        gp["roll_avg_to"]  = gp["to"].rolling(ROLL_WINDOW, 1).mean()

        # Lag stats
        gp["lag1_pts"] = gp["pts"].shift(1)
        gp["lag1_reb"] = gp["reb"].shift(1)
        gp["lag1_ast"] = gp["ast"].shift(1)
        gp["lag1_stl"] = gp["stl"].shift(1)
        gp["lag1_blk"] = gp["blk"].shift(1)
        gp["lag1_to"]  = gp["to"].shift(1)

        # Usage rate proxy
        gp["usage_proxy"] = (gp["fga"] + 0.44 * gp["fta"] + gp["to"]) / safe_min
        gp["lag1_usage"] = gp["usage_proxy"].shift(1)
        gp["roll_usage"] = gp["usage_proxy"].rolling(ROLL_WINDOW, 1).mean()
        gp["ewm_usage"] = gp["usage_proxy"].ewm(span=5, adjust=False).mean()

        # Games played
        gp["games_played"] = gp.groupby("player_id").cumcount()

        # Targets (next game)
        gp["target_pts"] = gp["pts"].shift(-1)
        gp["target_reb"] = gp["reb"].shift(-1)
        gp["target_ast"] = gp["ast"].shift(-1)
        gp["target_stl"] = gp["stl"].shift(-1)
        gp["target_blk"] = gp["blk"].shift(-1)
        gp["target_to"]  = gp["to"].shift(-1)

        feats.append(gp)

    return pd.concat(feats)


# ============================
# FINALIZE DATASET
# ============================
def finalize(df):
    keep_cols = [
        "player_id", "player", "team", "opponent", "season", "date",
        "pts", "reb", "ast", "stl", "blk", "to",
        "min", "is_home",

        "roll_avg_pts", "roll_avg_reb", "roll_avg_ast",
        "roll_avg_stl", "roll_avg_blk", "roll_avg_to",

        "lag1_pts", "lag1_reb", "lag1_ast",
        "lag1_stl", "lag1_blk", "lag1_to",

        "usage_proxy", "lag1_usage", "roll_usage", "ewm_usage",

        "opp_allow_pts", "opp_allow_reb", "opp_allow_ast",
        "opp_allow_stl", "opp_allow_blk", "opp_allow_to",
        "opp_pace_proxy",

        "games_played",

        "target_pts", "target_reb", "target_ast",
        "target_stl", "target_blk", "target_to",
    ]

    df = df[keep_cols].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved training dataset → {OUT}")
    print("Final shape:", df.shape)


# ============================
# MAIN PIPELINE
# ============================
def main():
    df = load_data()
    df = add_opponent_context(df)
    id_df = get_identity(df)

    df = df.merge(
        id_df[["player_id", "player", "team"]],
        on="player_id",
        how="inner"
    )

    # ✅ Restore team column after merge
    if "team_x" in df.columns:
        df["team"] = df["team_x"]
    elif "team_y" in df.columns:
        df["team"] = df["team_y"]

    for bad in ["team_x", "team_y"]:
        if bad in df.columns:
            df.drop(columns=[bad], inplace=True)

    df = build_features(df)
    finalize(df)


if __name__ == "__main__":
    main()