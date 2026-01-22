import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse

SETTLED_DIR = Path("data/history/settled")

BASE_DIR = Path("data/bankroll_builder")
SLIPS_DIR = BASE_DIR / "daily_slips"
RESULTS_DIR = BASE_DIR / "daily_results"
STATS_DIR = BASE_DIR / "daily_stats"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

PERF_PATH = BASE_DIR / "daily_performance.csv"

# Same flex payouts as builder
PAYOUT_3OF3_MULT = 3.0
PAYOUT_2OF3_MULT = 1.0
PAYOUT_1OF3_MULT = 0.0
PAYOUT_0OF3_MULT = 0.0

EPS = 1e-9

def bucket_edge(e: float) -> str:
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

def bankroll_start_for_date(target_date: str, default_initial=200.0) -> float:
    if not PERF_PATH.exists():
        return float(default_initial)
    df = pd.read_csv(PERF_PATH)
    if df.empty:
        return float(default_initial)

    # if date already exists, take that bankroll_start (regrade-safe)
    match = df[df["date"] == target_date]
    if not match.empty:
        return float(match.iloc[0]["bankroll_start"])

    # otherwise last bankroll_end
    return float(df.iloc[-1].get("bankroll_end", default_initial))

def payout_mult(wins: int) -> float:
    if wins == 3:
        return PAYOUT_3OF3_MULT
    if wins == 2:
        return PAYOUT_2OF3_MULT
    if wins == 1:
        return PAYOUT_1OF3_MULT
    return PAYOUT_0OF3_MULT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
    args = parser.parse_args()
    target_date = args.date

    slips_path = SLIPS_DIR / f"builder_slips_{target_date}.csv"
    if not slips_path.exists():
        raise FileNotFoundError(f"Missing slips file: {slips_path}")

    settled_path = SETTLED_DIR / f"settled_{target_date}.csv"
    if not settled_path.exists():
        raise FileNotFoundError(f"Missing settled file: {settled_path}")

    print(f"\n📥 Loading slips: {slips_path}")
    slips = pd.read_csv(slips_path)

    print(f"📥 Loading settled: {settled_path}")
    settled = pd.read_csv(settled_path)

    required = {"player", "prop", "result"}
    missing = required - set(settled.columns)
    if missing:
        raise ValueError(f"Settled missing required columns: {missing}")

    # normalize
    settled["player"] = settled["player"].astype(str).str.strip()
    settled["prop"] = settled["prop"].astype(str).str.strip().str.upper()
    settled["result"] = settled["result"].astype(str).str.strip().str.upper()

    # map results per leg
    settled["key"] = settled["player"] + "|" + settled["prop"]
    result_map = dict(zip(settled["key"], settled["result"]))

    # grade slips
    rows = []
    total_return = 0.0
    total_staked = 0.0

    for _, r in slips.iterrows():
        players = str(r["players"]).split(" | ")
        props = str(r["props"]).split(" | ")
        slip_size = float(r["slip_size"])

        leg_results = []
        wins = 0
        losses = 0
        pushes = 0
        missing = 0

        for p, pr in zip(players, props):
            k = f"{p.strip()}|{pr.strip().upper()}"
            res = result_map.get(k, "MISSING")
            leg_results.append(res)

            if res == "WIN":
                wins += 1
            elif res == "LOSS":
                losses += 1
            elif res == "PUSH":
                pushes += 1
            else:
                missing += 1

        # treat PUSH as neutral; for prizepicks it usually voids leg behavior varies.
        # Here: if any missing -> mark slip missing so it won’t affect bankroll
        if missing > 0:
            slip_outcome = "MISSING"
            mult = 0.0
            slip_return = 0.0
            profit = 0.0
        else:
            # for this simulator: PUSH counts as "not win" (reduces win count)
            # you can change this policy later if you want “push = refund that leg”
            effective_wins = wins
            slip_outcome = f"{effective_wins}/3"

            mult = payout_mult(effective_wins)
            slip_return = slip_size * mult
            profit = slip_return - slip_size

            total_staked += slip_size
            total_return += slip_return

        rows.append({
            **r.to_dict(),
            "legs_results": " | ".join(leg_results),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "missing_legs": missing,
            "slip_outcome": slip_outcome,
            "payout_mult": mult,
            "slip_return": round(slip_return, 2),
            "profit": round(profit, 2),
        })

    results = pd.DataFrame(rows)

    out_path = RESULTS_DIR / f"builder_results_{target_date}.csv"
    results.to_csv(out_path, index=False)

    print(f"\n✅ Saved daily bankroll results → {out_path}")

    # daily parlay stats by legs + edge bucket (edge bucket based on avg leg edge_for_pick)
    if "true_edges_for_pick" in results.columns:
        # parse average edge from "a | b | c"
        def avg_edge(s):
            try:
                parts = [float(x.strip()) for x in str(s).split("|")]
                if not parts:
                    return np.nan
                return float(np.mean(parts))
            except:
                return np.nan

        results["avg_leg_edge_for_pick"] = results["true_edges_for_pick"].apply(avg_edge)
        results["edge_bucket"] = results["avg_leg_edge_for_pick"].apply(bucket_edge)
    else:
        results["avg_leg_edge_for_pick"] = np.nan
        results["edge_bucket"] = "unknown"

    # per-day breakdown
    day_stats = (
        results[results["slip_outcome"] != "MISSING"]
        .groupby(["legs", "edge_bucket"], as_index=False)
        .agg(
            slips=("slip_id", "count"),
            avg_ev=("expected_value", "mean"),
            avg_edge=("avg_leg_edge_for_pick", "mean"),
            wins_3of3=("wins", lambda x: int((x == 3).sum())),
            wins_2of3=("wins", lambda x: int((x == 2).sum())),
            wins_1of3=("wins", lambda x: int((x == 1).sum())),
            wins_0of3=("wins", lambda x: int((x == 0).sum())),
            total_profit=("profit", "sum"),
        )
    )

    stats_path = STATS_DIR / f"builder_parlay_stats_{target_date}.csv"
    day_stats.to_csv(stats_path, index=False)
    print(f"✅ Saved daily parlay stats → {stats_path}")

    # print quick summary
    graded = results[results["slip_outcome"] != "MISSING"].copy()
    if graded.empty:
        print("⚠️ No graded slips (all missing).")
        return

    print("\n📊 Daily slip outcomes:")
    print(graded["slip_outcome"].value_counts().to_string())

    print("\n💵 Daily P&L:")
    print(f" - Total staked: ${total_staked:.2f}")
    print(f" - Total return: ${total_return:.2f}")
    print(f" - Profit: ${total_return - total_staked:.2f}")
    if total_staked > 0:
        print(f" - ROI: {(total_return - total_staked) / total_staked:.4f}")


if __name__ == "__main__":
    main()