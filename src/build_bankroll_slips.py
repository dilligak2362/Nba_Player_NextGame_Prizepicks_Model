import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import itertools

# -----------------------
# PATHS
# -----------------------
BOARD_PATH = Path("data/processed/calibrated_board.csv")

OUT_DIR = Path("data/bankroll_builder")
SLIPS_DIR = OUT_DIR / "daily_slips"
SLIPS_DIR.mkdir(parents=True, exist_ok=True)

PERF_PATH = OUT_DIR / "daily_performance.csv"

# -----------------------
# CONFIG
# -----------------------
INITIAL_BANKROLL = 200.0
DAILY_RISK_PCT = 0.30

LEGS = 3
TARGET_SLIPS = 6

# Starting filters (script will auto-relax if it cannot fill 6 slips)
MIN_PROP_PROB_START = 0.52
TOP_POOL_SIZE_START = 200

# Try these relax steps in order until we can build all slips
# (prob threshold, pool size)
RELAX_STEPS = [
    (0.52, 200),
    (0.515, 250),
    (0.51, 300),
    (0.505, 350),
    (0.50, 450),
]

# To keep runtime fast and prevent hangs
# We do NOT brute-force all combos of the full pool.
CANDIDATES_PER_SLIP = 45          # how many top "available" legs to consider per slip
COMBOS_PER_SLIP_CAP = 20000       # cap combos evaluated per slip (safe speed)

# Adds randomness so you don't get the same exact slate every run
RANDOM_SEED = None                # set to an int for reproducible results, e.g. 42
RANDOMNESS_STRENGTH = 0.12        # 0 = purely deterministic, higher = more variety

# Optional: prefer not-all-same direction inside a slip
PREFER_MIXED_DIRECTIONS = True

# -----------------------
# HELPERS
# -----------------------

def bankroll_start_for_date(target_date, default=INITIAL_BANKROLL):
    if not PERF_PATH.exists():
        return default
    df = pd.read_csv(PERF_PATH)
    if df.empty:
        return default
    return float(df.iloc[-1].get("bankroll_end", default))


def compute_flex_ev(p1, p2, p3, stake):
    p1, p2, p3 = [float(np.clip(x, 0, 1)) for x in (p1, p2, p3)]
    p3_hit = p1 * p2 * p3
    p2_hit = (
        p1 * p2 * (1 - p3)
        + p1 * p3 * (1 - p2)
        + p2 * p3 * (1 - p1)
    )
    return stake * (3 * p3_hit + p2_hit) - stake


def normalize_probability_column(df):
    # If board already has a probability column, standardize it to p_leg
    for c in ["p_leg", "p_leg_calibrated", "p_calibrated", "prob", "p", "win_prob"]:
        if c in df.columns:
            return df.rename(columns={c: "p_leg"}) if c != "p_leg" else df

    # Otherwise derive from edge
    if "true_edge_for_pick" not in df.columns:
        raise RuntimeError("No probability column found and cannot derive from missing `true_edge_for_pick`.")

    print("⚠️ No probability column found — deriving p_leg from true_edge_for_pick")
    df["p_leg"] = 0.5 + np.tanh(df["true_edge_for_pick"] / 2.0) * 0.08
    df["p_leg"] = df["p_leg"].clip(0.40, 0.65)
    return df


def _safe_line_value(x):
    # used for leg uniqueness; stable even if line missing
    try:
        return float(x)
    except:
        return np.nan


def prepare_board(board: pd.DataFrame) -> pd.DataFrame:
    board = normalize_probability_column(board)

    df = board.copy()

    # Normalize / clean
    for c in ["player", "team", "prop", "direction"]:
        if c not in df.columns:
            raise RuntimeError(f"Board missing required column: {c}")

    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.strip()
    df["prop"] = df["prop"].astype(str).str.strip().str.upper()
    df["direction"] = df["direction"].astype(str).str.strip().str.upper()

    df["p_leg"] = pd.to_numeric(df["p_leg"], errors="coerce")

    if "true_edge_for_pick" not in df.columns:
        raise RuntimeError("Board missing required column: true_edge_for_pick")
    df["true_edge_for_pick"] = pd.to_numeric(df["true_edge_for_pick"], errors="coerce")

    # book_line is optional, but if present we include it in uniqueness key
    if "book_line" in df.columns:
        df["book_line"] = df["book_line"].apply(_safe_line_value)
        df["book_line_key"] = df["book_line"].round(3).astype(str)
    else:
        df["book_line_key"] = "NA"

    # Drop garbage rows
    df = df.dropna(subset=["player", "team", "prop", "direction", "p_leg", "true_edge_for_pick"]).copy()

    # Global leg key = PLAYER + PROP + DIRECTION + LINE
    # This makes "Keyonte George UNDER PTS" unique (and blocks repeats).
    df["leg_key"] = (
        df["player"] + "|" +
        df["prop"] + "|" +
        df["direction"] + "|" +
        df["book_line_key"]
    )

    # Score for ranking (mostly prob + edge, with small noise to avoid “same slate every time”)
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(0, 1, size=len(df)) * RANDOMNESS_STRENGTH
    df["rank_score"] = (df["p_leg"] * 10.0) + (df["true_edge_for_pick"] * 1.0) + noise

    return df


def best_slip_from_available(available_df: pd.DataFrame, slip_size: float) -> dict | None:
    """
    Pick best 3-leg combination from available legs, enforcing:
    - unique player inside slip
    - unique team inside slip
    - unique prop type inside slip (PTS / PRA / etc) inside slip
    """
    if len(available_df) < LEGS:
        return None

    # Take top candidates for this slip
    cand = available_df.sort_values("rank_score", ascending=False).head(CANDIDATES_PER_SLIP).copy()
    if len(cand) < LEGS:
        return None

    idxs = cand.index.to_list()

    best = None
    combos_eval = 0

    # Evaluate combos but cap them
    for combo in itertools.combinations(idxs, LEGS):
        combos_eval += 1
        if combos_eval > COMBOS_PER_SLIP_CAP:
            break

        rows = cand.loc[list(combo)]

        # within-slip uniqueness
        if rows["player"].nunique() != LEGS:
            continue
        if rows["team"].nunique() != LEGS:
            continue
        if rows["prop"].nunique() != LEGS:
            continue

        # Optional preference: avoid all 3 being OVER or all 3 being UNDER
        if PREFER_MIXED_DIRECTIONS:
            if rows["direction"].nunique() == 1:
                # not forbidden, but penalize slightly so mixed slips are favored
                direction_penalty = 0.01
            else:
                direction_penalty = 0.0
        else:
            direction_penalty = 0.0

        ev = compute_flex_ev(
            rows.iloc[0]["p_leg"],
            rows.iloc[1]["p_leg"],
            rows.iloc[2]["p_leg"],
            slip_size
        ) - direction_penalty

        if (best is None) or (ev > best["ev"]):
            best = {"rows": rows, "ev": ev}

    return best


# -----------------------
# CORE BUILDER
# -----------------------

def build_slips(board: pd.DataFrame, bankroll_start: float, target_date: str) -> pd.DataFrame:
    daily_risk = bankroll_start * DAILY_RISK_PCT
    slip_size = daily_risk / TARGET_SLIPS

    df = prepare_board(board)

    # Try relax steps until we can fill TARGET_SLIPS with GLOBAL UNIQUE legs
    for (min_prob, top_pool) in RELAX_STEPS:
        pool = df[df["p_leg"] >= min_prob].copy()
        pool = pool.sort_values("rank_score", ascending=False).head(top_pool).copy()

        # Quick feasibility check: do we even have enough unique legs?
        needed_legs = TARGET_SLIPS * LEGS
        unique_legs = pool["leg_key"].nunique()
        if unique_legs < needed_legs:
            print(f"⚠️ Relax step (p>={min_prob}, pool={top_pool}) has only {unique_legs} unique legs; need {needed_legs}.")
            continue

        used_legs = set()
        slips = []

        # Build slips sequentially, blocking ANY reused player-prop-direction-line
        for slip_num in range(1, TARGET_SLIPS + 1):
            available = pool[~pool["leg_key"].isin(used_legs)].copy()

            pick = best_slip_from_available(available, slip_size)
            if pick is None:
                slips = []
                break

            rows = pick["rows"]
            ev = pick["ev"]

            # Mark legs used globally
            for lk in rows["leg_key"].tolist():
                used_legs.add(lk)

            slips.append({
                "date": target_date,
                "slip_id": slip_num,
                "bankroll_start": round(bankroll_start, 2),
                "daily_risk": round(daily_risk, 2),
                "slip_size": round(slip_size, 2),
                "legs": LEGS,
                "players": " | ".join(rows["player"].tolist()),
                "teams": " | ".join(rows["team"].tolist()),
                "props": " | ".join(rows["prop"].tolist()),
                "directions": " | ".join(rows["direction"].tolist()),
                "p_legs": " | ".join(f"{x:.3f}" for x in rows["p_leg"].tolist()),
                "expected_value": round(ev, 4),
            })

        if len(slips) == TARGET_SLIPS:
            out_df = pd.DataFrame(slips).sort_values("expected_value", ascending=False).reset_index(drop=True)
            out_df["slip_id"] = np.arange(1, len(out_df) + 1)

            print(f"✅ Built {TARGET_SLIPS} slips with GLOBAL UNIQUE legs using min_prob={min_prob}, pool={top_pool}")
            return out_df

        print(f"⚠️ Relax step (p>={min_prob}, pool={top_pool}) could not complete all slips. Trying next step...")

    raise RuntimeError(
        "Could not build 6 slips with globally-unique player props. "
        "Your pool is too small at current thresholds. "
        "Try lowering MIN_PROP_PROB_START, increasing RELAX_STEPS pool sizes, "
        "or loosen within-slip constraints (like prop uniqueness inside slip)."
    )


# -----------------------
# MAIN
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    board = pd.read_csv(BOARD_PATH)
    bankroll_start = bankroll_start_for_date(args.date)

    print(f"\n💰 Bankroll: ${bankroll_start:.2f}")
    print(f"🎯 Target slips: {TARGET_SLIPS}")

    slips = build_slips(board, bankroll_start, args.date)

    out = SLIPS_DIR / f"builder_slips_{args.date}.csv"
    slips.to_csv(out, index=False)

    print(f"\n✅ Saved {len(slips)} slips → {out}")
    print(slips[["slip_id", "players", "props", "directions", "expected_value"]].to_string(index=False))


if __name__ == "__main__":
    main()