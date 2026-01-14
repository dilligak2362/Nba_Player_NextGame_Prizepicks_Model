import pandas as pd
import numpy as np
from pathlib import Path

SETTLED_DIR = Path("data/history/settled")
OUT_DIR = Path("data/performance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    files = sorted(SETTLED_DIR.glob("settled_*.csv"))
    if not files:
        print("❌ No settled files found yet.")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    required = {"result", "edge", "prop", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in settled files: {missing}")

    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")

    df = df[df["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    if df.empty:
        print("⚠️ No gradeable results found.")
        return

    df["is_win"] = (df["result"] == "WIN").astype(int)
    df["is_loss"] = (df["result"] == "LOSS").astype(int)
    df["is_push"] = (df["result"] == "PUSH").astype(int)

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
        "avg_edge": round(df["edge"].mean(), 4),
    }])

    summary_path = OUT_DIR / "performance_summary.csv"
    summary.to_csv(summary_path, index=False)

    # ===========================
    # BY PROP TYPE
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

    by_prop_path = OUT_DIR / "by_prop.csv"
    by_prop.to_csv(by_prop_path, index=False)

    # ===========================
    # BY DIRECTION (OVER vs UNDER)
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

    by_direction_path = OUT_DIR / "by_direction.csv"
    by_direction.to_csv(by_direction_path, index=False)

   # ===========================
# BY EDGE BUCKET (ULTRA GRANULAR)
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

    # order buckets logically
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

    by_edge_path = OUT_DIR / "by_edge_bucket.csv"
    by_edge.to_csv(by_edge_path, index=False)

    # ===========================
    # OUTPUT
    # ===========================
    print("\n✅ Performance reports saved:")
    print(f" - {summary_path}")
    print(f" - {by_prop_path}")
    print(f" - {by_direction_path}")
    print(f" - {by_edge_path}")

if __name__ == "__main__":
    main()