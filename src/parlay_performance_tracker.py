import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ===========================
# PATHS
# ===========================
PARLAY_RESULTS_PATH = Path("data/performance/parlay_results.csv")

# nightly settlement folder
PARLAY_SETTLED_DIR = Path("data/parlay_results")
PARLAY_SETTLED_DIR.mkdir(parents=True, exist_ok=True)

# analytics folder (global)
OUT_DIR = Path("data/parlay_performance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# daily analytics folder
DAILY_STATS_DIR = Path("data/parlay_daily_stats")
DAILY_STATS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "parlay_summary.csv"
BY_LEGS_PATH = OUT_DIR / "parlay_by_legs.csv"
BY_EDGE_BUCKET_LEGS_PATH = OUT_DIR / "parlay_by_edge_bucket_and_legs.csv"
DETAILED_RESULTS_PATH = OUT_DIR / "parlay_results_detailed.csv"


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


# ===========================
# MAIN
# ===========================
def main(target_date: str = None):
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if not PARLAY_RESULTS_PATH.exists():
        print("❌ No parlay results found. Run performance_tracker.py first.")
        return

    print("\n📥 Loading parlay results...")
    df = pd.read_csv(PARLAY_RESULTS_PATH)

    required = {"legs", "total_edge", "result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["total_edge"] = pd.to_numeric(df["total_edge"], errors="coerce")

    df = df[df["result"].isin(["WIN", "LOSS", "PUSH"])].copy()

    df["is_win"] = (df["result"] == "WIN").astype(int)
    df["is_loss"] = (df["result"] == "LOSS").astype(int)
    df["is_push"] = (df["result"] == "PUSH").astype(int)

    # ===========================
    # SAVE NIGHTLY PARLAY SETTLEMENT
    # ===========================
    settled_path = PARLAY_SETTLED_DIR / f"settled_parlays_{target_date}.csv"
    df.to_csv(settled_path, index=False)
    print(f"\n📒 Saved nightly parlay settlement → {settled_path}")

    # ===========================
    # EDGE BUCKET
    # ===========================
    df["edge_bucket"] = df["total_edge"].apply(bucket_edge)

    # ===========================
    # DAILY EDGE BUCKET × LEGS REPORT
    # ===========================
    daily_edge_legs = (
        df.groupby(["legs", "edge_bucket"], as_index=False)
        .agg(
            slips=("edge_bucket", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_total_edge=("total_edge", "mean"),
        )
    )

    daily_edge_legs["win_rate_ex_push"] = (
        daily_edge_legs["wins"] /
        (daily_edge_legs["wins"] + daily_edge_legs["losses"]).replace(0, np.nan)
    )

    daily_path = DAILY_STATS_DIR / f"parlay_edge_legs_{target_date}.csv"
    daily_edge_legs.to_csv(daily_path, index=False)

    print(f"\n📊 Saved daily parlay edge+legs stats → {daily_path}")

    # ===========================
    # GLOBAL SUMMARY
    # ===========================
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
        "avg_total_edge": round(df["total_edge"].mean(), 4),
    }])

    summary.to_csv(SUMMARY_PATH, index=False)

    # ===========================
    # BY LEGS (GLOBAL)
    # ===========================
    by_legs = (
        df.groupby("legs", as_index=False)
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

    by_legs.to_csv(BY_LEGS_PATH, index=False)

    # ===========================
    # BY EDGE BUCKET + LEGS (GLOBAL)
    # ===========================
    by_edge_legs = (
        df.groupby(["legs", "edge_bucket"], as_index=False)
        .agg(
            slips=("edge_bucket", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_total_edge=("total_edge", "mean"),
        )
    )

    by_edge_legs["win_rate_ex_push"] = (
        by_edge_legs["wins"] / (by_edge_legs["wins"] + by_edge_legs["losses"]).replace(0, np.nan)
    )

    by_edge_legs.to_csv(BY_EDGE_BUCKET_LEGS_PATH, index=False)

    # ===========================
    # SAVE FULL HISTORY
    # ===========================
    df.to_csv(DETAILED_RESULTS_PATH, index=False)

    # ===========================
    # OUTPUT
    # ===========================
    print("\n✅ Parlay performance reports saved:")
    print(f" - {SUMMARY_PATH}")
    print(f" - {BY_LEGS_PATH}")
    print(f" - {BY_EDGE_BUCKET_LEGS_PATH}")
    print(f" - {DETAILED_RESULTS_PATH}")
    print(f" - {daily_path}")

    print("\n🔥 Daily Parlay Summary:")
    print(daily_edge_legs)


if __name__ == "__main__":
    main()