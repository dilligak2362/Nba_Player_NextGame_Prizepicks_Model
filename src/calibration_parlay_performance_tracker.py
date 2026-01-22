import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PARLAY_DIR = Path("data/calibration_parlays")
PERF_DIR = Path("data/calibration_parlay_performance")
DAILY_DIR = Path("data/calibration_parlay_daily_stats")

PERF_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# EDGE BUCKET
# --------------------------
def bucket_edge(e):
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


def main(target_date=None):
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    settled_path = PARLAY_DIR / f"calibration_settled_parlays_{target_date}.csv"

    if not settled_path.exists():
        raise FileNotFoundError(f"Missing {settled_path}")

    print("\n📥 Loading settled calibration parlays...")
    df = pd.read_csv(settled_path)

    df["total_edge"] = pd.to_numeric(df["total_edge"], errors="coerce")

    df = df[df["result"].isin(["WIN", "LOSS", "PUSH"])].copy()

    df["is_win"] = (df["result"] == "WIN").astype(int)
    df["is_loss"] = (df["result"] == "LOSS").astype(int)
    df["is_push"] = (df["result"] == "PUSH").astype(int)

    df["edge_bucket"] = df["total_edge"].apply(bucket_edge)

    # --------------------------
    # DAILY EDGE × LEGS
    # --------------------------
    daily = (
        df.groupby(["legs", "edge_bucket"], as_index=False)
        .agg(
            slips=("edge_bucket", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_edge=("total_edge", "mean"),
        )
    )

    daily["win_rate_ex_push"] = (
        daily["wins"] /
        (daily["wins"] + daily["losses"]).replace(0, np.nan)
    )

    daily_path = DAILY_DIR / f"calibration_parlay_edge_legs_{target_date}.csv"
    daily.to_csv(daily_path, index=False)

    # --------------------------
    # GLOBAL PERFORMANCE
    # --------------------------
    df.to_csv(PERF_DIR / "parlay_results_detailed.csv", index=False)

    by_legs = (
        df.groupby("legs", as_index=False)
        .agg(
            slips=("legs", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_edge=("total_edge", "mean"),
        )
    )

    by_legs["win_rate_ex_push"] = (
        by_legs["wins"] /
        (by_legs["wins"] + by_legs["losses"]).replace(0, np.nan)
    )

    by_legs.to_csv(PERF_DIR / "parlay_by_legs.csv", index=False)

    by_edge_legs = (
        df.groupby(["legs", "edge_bucket"], as_index=False)
        .agg(
            slips=("edge_bucket", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_edge=("total_edge", "mean"),
        )
    )

    by_edge_legs["win_rate_ex_push"] = (
        by_edge_legs["wins"] /
        (by_edge_legs["wins"] + by_edge_legs["losses"]).replace(0, np.nan)
    )

    by_edge_legs.to_csv(PERF_DIR / "parlay_by_edge_bucket_and_legs.csv", index=False)

    print(f"\n✅ Calibration parlay performance saved")
    print(f"📊 Daily stats → {daily_path}")


if __name__ == "__main__":
    main()