import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# -----------------------
# PATHS
# -----------------------
BOARD_PATH = Path("data/processed/calibrated_board.csv")
SETTLED_DIR = Path("data/history/settled")

OUT_DIR = Path("data/bankroll_builder")
SLIPS_DIR = OUT_DIR / "daily_slips"
SLIPS_DIR.mkdir(parents=True, exist_ok=True)

PERF_PATH = OUT_DIR / "daily_performance.csv"

# -----------------------
# CONFIG
# -----------------------
INITIAL_BANKROLL = 200.0
DAILY_RISK_PCT = 0.30

MIN_SLIP_SIZE = 5.0
LEGS = 3

PAYOUT_3OF3_MULT = 3.0
PAYOUT_2OF3_MULT = 1.0
PAYOUT_1OF3_MULT = 0.0
PAYOUT_0OF3_MULT = 0.0

MIN_PROP_SAMPLES_IN_BUCKET = 20
MIN_PROP_PROB = 0.52
MAX_SLIPS_PER_PLAYER = 2
MAX_SLIPS_PER_TEAM = 3

NO_DUP_PLAYER_IN_SLIP = True
NO_DUP_TEAM_IN_SLIP = True

TOP_POOL_SIZE = 120
EPS = 1e-9

# -----------------------
# HELPERS
# -----------------------

def bankroll_start_for_date(target_date: str, default_initial=INITIAL_BANKROLL) -> float:
    if not PERF_PATH.exists():
        return float(default_initial)

    df = pd.read_csv(PERF_PATH)
    if df.empty:
        return float(default_initial)

    match = df[df["date"] == target_date]
    if not match.empty:
        return float(match.iloc[0]["bankroll_start"])

    return float(df.iloc[-1].get("bankroll_end", default_initial))


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


def compute_flex_ev(p1, p2, p3, stake: float) -> float:
    p1, p2, p3 = map(lambda x: float(np.clip(x, 0, 1)), [p1, p2, p3])

    p_3 = p1 * p2 * p3
    p_2 = (p1*p2*(1-p3)) + (p1*p3*(1-p2)) + (p2*p3*(1-p1))

    exp_return = stake * (
        p_3 * PAYOUT_3OF3_MULT +
        p_2 * PAYOUT_2OF3_MULT
    )

    return exp_return - stake


def learn_bucket_winrates():
    files = sorted(SETTLED_DIR.glob("settled_*.csv"))
    if not files:
        return pd.DataFrame()

    hist = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    if not {"prop", "direction", "result"}.issubset(hist.columns):
        return pd.DataFrame()

    if "true_edge_for_pick" in hist.columns:
        edge_col = "true_edge_for_pick"
    elif "true_edge" in hist.columns:
        edge_col = "true_edge"
    elif "edge" in hist.columns:
        edge_col = "edge"
    else:
        return pd.DataFrame()

    hist = hist[hist["result"].isin(["WIN", "LOSS"])].copy()
    hist["is_win"] = (hist["result"] == "WIN").astype(int)
    hist["edge_bucket"] = hist[edge_col].apply(bucket_edge)

    g = hist.groupby(["prop", "direction", "edge_bucket"], as_index=False).agg(
        n=("prop", "count"),
        wins=("is_win", "sum"),
    )
    g["win_rate"] = g["wins"] / g["n"]

    return g


def attach_probabilities(board, winrates):
    df = board.copy()
    df["direction"] = df["direction"].str.upper()

    df["true_edge_for_pick"] = np.where(
        df["direction"] == "UNDER",
        df["book_line"] - df["model_prediction"],
        df["model_prediction"] - df["book_line"]
    )

    df["edge_bucket"] = df["true_edge_for_pick"].apply(bucket_edge)

    if winrates.empty:
        df["p_leg"] = 0.50 + np.tanh(df["true_edge_for_pick"] / 2) * 0.08
        df["p_leg"] = df["p_leg"].clip(0.40, 0.65)
        return df

    df = df.merge(winrates, on=["prop", "direction", "edge_bucket"], how="left")
    df["p_leg"] = df["win_rate"].fillna(0.50).clip(0.35, 0.75)

    return df


def build_slips(df, bankroll_start, target_date):
    daily_risk = bankroll_start * DAILY_RISK_PCT
    slips_target = int(np.floor(daily_risk / MIN_SLIP_SIZE))
    slip_size = daily_risk / slips_target

    df = df[df["p_leg"] >= MIN_PROP_PROB].copy()
    df = df.sort_values(["p_leg", "true_edge_for_pick"], ascending=False).head(TOP_POOL_SIZE)

    slips = []
    slip_id = 1
    player_use, team_use = {}, {}

    for i in range(len(df)):
        if len(slips) >= slips_target:
            break

        a = df.iloc[i]

        if player_use.get(a["player"], 0) >= MAX_SLIPS_PER_PLAYER:
            continue
        if team_use.get(a["team"], 0) >= MAX_SLIPS_PER_TEAM:
            continue

        comps = []
        for j in range(i+1, len(df)):
            b = df.iloc[j]
            if b["player"] == a["player"] or b["team"] == a["team"]:
                continue
            comps.append(b)

        if len(comps) < 2:
            continue

        best = None
        for x in range(len(comps)):
            for y in range(x+1, len(comps)):
                b, c = comps[x], comps[y]
                ev = compute_flex_ev(a["p_leg"], b["p_leg"], c["p_leg"], slip_size)
                if best is None or ev > best["ev"]:
                    best = {"a": a, "b": b, "c": c, "ev": ev}

        if not best:
            continue

        legs = [best["a"], best["b"], best["c"]]

        slips.append({
            "date": target_date,
            "slip_id": slip_id,
            "bankroll_start": round(bankroll_start, 2),
            "daily_risk": round(daily_risk, 2),
            "slip_size": round(slip_size, 2),
            "players": " | ".join(x["player"] for x in legs),
            "teams": " | ".join(x["team"] for x in legs),
            "props": " | ".join(x["prop"] for x in legs),
            "directions": " | ".join(x["direction"] for x in legs),
            "p_legs": " | ".join(str(round(x["p_leg"], 4)) for x in legs),
            "expected_value": round(best["ev"], 4)
        })

        for x in legs:
            player_use[x["player"]] = player_use.get(x["player"], 0) + 1
            team_use[x["team"]] = team_use.get(x["team"], 0) + 1

        slip_id += 1

    return pd.DataFrame(slips)


# -----------------------
# MAIN
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    target_date = args.date

    board = pd.read_csv(BOARD_PATH)
    bankroll_start = bankroll_start_for_date(target_date)

    print(f"\n💰 Bankroll start = ${bankroll_start:.2f}")
    print(f"🎯 Daily risk = ${bankroll_start * DAILY_RISK_PCT:.2f}")

    winrates = learn_bucket_winrates()
    board = attach_probabilities(board, winrates)

    slips = build_slips(board, bankroll_start, target_date)

    out_path = SLIPS_DIR / f"builder_slips_{target_date}.csv"
    slips.to_csv(out_path, index=False)

    print(f"\n✅ Saved slips → {out_path}")
    print(f"Total slips: {len(slips)}")
    print(slips.head(10).to_string(index=False))


if __name__ == "__main__":
    main()