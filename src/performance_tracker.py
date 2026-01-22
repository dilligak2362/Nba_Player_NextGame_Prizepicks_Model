import pandas as pd
import numpy as np
from pathlib import Path

SETTLED_DIR = Path("data/history/settled")
PARLAY_SLIPS_PATH = Path("data/processed/parlay_slips.csv")

OUT_DIR = Path("data/performance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "performance_summary.csv"
BY_PROP_PATH = OUT_DIR / "by_prop.csv"
BY_DIRECTION_PATH = OUT_DIR / "by_direction.csv"
BY_EDGE_BUCKET_PATH = OUT_DIR / "by_edge_bucket.csv"

PARLAY_RESULTS_PATH = OUT_DIR / "parlay_results.csv"
PARLAY_STATS_PATH = OUT_DIR / "parlay_performance.csv"


# ===========================
# EDGE BUCKETS
# ===========================
EPS = 1e-9

def bucket_edge(e):
    if pd.isna(e):
        return "unknown"

    if abs(e) <= EPS:
        return "=0"

    if e < -5: return "<-5"
    if e < -3: return "-5 to -3"
    if e < -2: return "-3 to -2"
    if e < -1: return "-2 to -1"
    if e < -0.25: return "-1 to -0.25"
    if e < 0: return "-0.25 to 0"

    if e <= 0.25: return "0 to 0.25"
    if e < 1: return "0.25 to 1"
    if e < 2: return "1 to 2"
    if e < 3: return "2 to 3"
    if e < 5: return "3 to 5"
    return "5+"


def main():
    # ===========================
    # Load settled props
    # ===========================
    files = sorted(SETTLED_DIR.glob("settled_*.csv"))
    if not files:
        print("❌ No settled files found yet.")
        return

    print("\n📥 Loading settled results...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    required = {"player", "prop", "direction", "edge", "result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in settled files: {missing}")

    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")
    df = df[df["result"].isin(["WIN", "LOSS", "PUSH"])].copy()

    df["is_win"] = (df["result"] == "WIN").astype(int)
    df["is_loss"] = (df["result"] == "LOSS").astype(int)
    df["is_push"] = (df["result"] == "PUSH").astype(int)

    # ===========================
    # GLOBAL SUMMARY
    # ===========================
    print("\n📊 Building global performance summary...")

    total = len(df)
    wins = df["is_win"].sum()
    losses = df["is_loss"].sum()
    pushes = df["is_push"].sum()

    win_rate = wins / max(1, (wins + losses))

    summary = pd.DataFrame([{
        "n_total": int(total),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "win_rate_ex_push": round(win_rate, 4),
        "avg_edge": round(df["edge"].mean(), 4),
    }])

    summary.to_csv(SUMMARY_PATH, index=False)

    # ===========================
    # BY PROP
    # ===========================
    by_prop = (
        df.groupby("prop", as_index=False)
        .agg(
            n=("prop", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_edge=("edge", "mean"),
        )
    )

    by_prop["win_rate_ex_push"] = (
        by_prop["wins"] / (by_prop["wins"] + by_prop["losses"]).replace(0, np.nan)
    )

    by_prop.to_csv(BY_PROP_PATH, index=False)

    # ===========================
    # BY DIRECTION
    # ===========================
    by_direction = (
        df.groupby("direction", as_index=False)
        .agg(
            n=("direction", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_edge=("edge", "mean"),
        )
    )

    by_direction["win_rate_ex_push"] = (
        by_direction["wins"] / (by_direction["wins"] + by_direction["losses"]).replace(0, np.nan)
    )

    by_direction.to_csv(BY_DIRECTION_PATH, index=False)

    # ===========================
    # BY EDGE BUCKET
    # ===========================
    df["edge_bucket"] = df["edge"].apply(bucket_edge)

    by_edge = (
        df.groupby("edge_bucket", as_index=False)
        .agg(
            n=("edge_bucket", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_edge=("edge", "mean"),
        )
    )

    by_edge["win_rate_ex_push"] = (
        by_edge["wins"] / (by_edge["wins"] + by_edge["losses"]).replace(0, np.nan)
    )

    bucket_order = [
        "<-5",
        "-5 to -3",
        "-3 to -2",
        "-2 to -1",
        "-1 to -0.25",
        "-0.25 to 0",
        "=0",
        "0 to 0.25",
        "0.25 to 1",
        "1 to 2",
        "2 to 3",
        "3 to 5",
        "5+",
        "unknown"
    ]

    by_edge["edge_bucket"] = pd.Categorical(
        by_edge["edge_bucket"],
        categories=bucket_order,
        ordered=True
    )

    by_edge = by_edge.sort_values("edge_bucket")
    by_edge.to_csv(BY_EDGE_BUCKET_PATH, index=False)

    # ===========================
    # PARLAY PERFORMANCE
    # ===========================
    print("\n🔗 Grading parlay slips...")

    if not PARLAY_SLIPS_PATH.exists():
        print("⚠️ No parlay slips found. Skipping parlay performance.")
        return

    parlays = pd.read_csv(PARLAY_SLIPS_PATH)

    df["key"] = df["player"] + "|" + df["prop"]
    result_map = dict(zip(df["key"], df["result"]))

    parlay_rows = []

    for _, row in parlays.iterrows():
        players = row["players"].split(" | ")
        props = row["props"].split(" | ")

        leg_results = []

        for p, pr in zip(players, props):
            key = f"{p}|{pr}"
            leg_results.append(result_map.get(key, "MISSING"))

        if "LOSS" in leg_results:
            final_result = "LOSS"
        elif "PUSH" in leg_results:
            final_result = "PUSH"
        elif "MISSING" in leg_results:
            final_result = "MISSING"
        else:
            final_result = "WIN"

        parlay_rows.append({
            **row,
            "result": final_result,
            "legs_results": " | ".join(leg_results)
        })

    parlay_results = pd.DataFrame(parlay_rows)

    parlay_results["is_win"] = (parlay_results["result"] == "WIN").astype(int)
    parlay_results["is_loss"] = (parlay_results["result"] == "LOSS").astype(int)
    parlay_results["is_push"] = (parlay_results["result"] == "PUSH").astype(int)

    parlay_results.to_csv(PARLAY_RESULTS_PATH, index=False)

    by_legs = (
        parlay_results.groupby("legs", as_index=False)
        .agg(
            slips=("legs", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_total_edge=("total_edge", "mean"),
        )
    )

    by_legs["win_rate_ex_push"] = (
        by_legs["wins"] / (by_legs["wins"] + by_legs["losses"]).replace(0, np.nan)
    )

    by_legs.to_csv(PARLAY_STATS_PATH, index=False)

    # ===========================
    # OUTPUT
    # ===========================
    print("\n✅ Performance reports saved:")
    print(f" - {SUMMARY_PATH}")
    print(f" - {BY_PROP_PATH}")
    print(f" - {BY_DIRECTION_PATH}")
    print(f" - {BY_EDGE_BUCKET_PATH}")
    print(f" - {PARLAY_RESULTS_PATH}")
    print(f" - {PARLAY_STATS_PATH}")

    print("\n🔥 Parlay Summary:")
    print(by_legs)


if __name__ == "__main__":
    main()