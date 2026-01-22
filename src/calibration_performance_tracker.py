import pandas as pd
import numpy as np
from pathlib import Path

# ===========================
# PATHS
# ===========================
CALIBRATION_RESULTS_DIR = Path("data/calibration_results")

OUT_DIR = Path("data/calibration_performance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "calibration_summary.csv"
BY_PROP_PATH = OUT_DIR / "calibration_by_prop.csv"
BY_DIRECTION_PATH = OUT_DIR / "calibration_by_direction.csv"
BY_EDGE_BUCKET_PATH = OUT_DIR / "calibration_by_edge_bucket.csv"


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
def main():
    files = sorted(CALIBRATION_RESULTS_DIR.glob("calibration_results_*.csv"))

    if not files:
        print("❌ No calibration result files found yet.")
        return

    print("\n📥 Loading calibration audit files...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    required = {"player", "prop", "direction", "true_edge_for_pick", "result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in calibration files: {missing}")

    df = df[df["result"].isin(["WIN", "LOSS", "PUSH"])].copy()

    df["true_edge_for_pick"] = pd.to_numeric(df["true_edge_for_pick"], errors="coerce")

    df["is_win"] = (df["result"] == "WIN").astype(int)
    df["is_loss"] = (df["result"] == "LOSS").astype(int)
    df["is_push"] = (df["result"] == "PUSH").astype(int)

    # ===========================
    # GLOBAL SUMMARY
    # ===========================
    print("\n📊 Building calibration performance summary...")

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
        "avg_true_edge_for_pick": round(df["true_edge_for_pick"].mean(), 4),
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
            avg_true_edge_for_pick=("true_edge_for_pick", "mean"),
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
            avg_true_edge_for_pick=("true_edge_for_pick", "mean"),
        )
    )

    by_direction["win_rate_ex_push"] = (
        by_direction["wins"] / (by_direction["wins"] + by_direction["losses"]).replace(0, np.nan)
    )

    by_direction.to_csv(BY_DIRECTION_PATH, index=False)

    # ===========================
    # BY EDGE BUCKET
    # ===========================
    df["edge_bucket"] = df["true_edge_for_pick"].apply(bucket_edge)

    by_edge = (
        df.groupby("edge_bucket", as_index=False)
        .agg(
            n=("edge_bucket", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            pushes=("is_push", "sum"),
            avg_true_edge_for_pick=("true_edge_for_pick", "mean"),
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
    # OUTPUT
    # ===========================
    print("\n✅ Calibration performance reports saved:")
    print(f" - {SUMMARY_PATH}")
    print(f" - {BY_PROP_PATH}")
    print(f" - {BY_DIRECTION_PATH}")
    print(f" - {BY_EDGE_BUCKET_PATH}")

    print("\n🔥 Calibration Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()