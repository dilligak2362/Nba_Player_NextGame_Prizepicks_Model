import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
BASE_URL = "https://api.balldontlie.io/v1/"
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")
HEADERS = {"Authorization": BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}

BOARDS_DIR = Path("data/history/boards")
SETTLED_DIR = Path("data/history/settled")
SETTLED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# HELPERS
# -----------------------
def safe_get(url, headers=None, params=None, tries=6):
    headers = headers or {}
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 + i
            print(f"Network/API issue: {e}. Retry {i+1}/{tries} in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"API failed too many times for {url}")


def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    for ch in [".", ",", "'", '"', "-", "’"]:
        s = s.replace(ch, "")
    return " ".join(s.split())


# -----------------------
# DATA FETCHING
# -----------------------
def get_games_for_date(date_str: str):
    if not BALLDONTLIE_API_KEY:
        raise RuntimeError("BALLDONTLIE_API_KEY not set.")

    games = []
    page = 1
    while True:
        data = safe_get(
            BASE_URL + "games",
            headers=HEADERS,
            params={"dates[]": date_str, "per_page": 100, "page": page},
        )
        games.extend(data.get("data", []))
        if page >= data.get("meta", {}).get("total_pages", 1):
            break
        page += 1
    return games


def get_stats_for_game(game_id: int) -> pd.DataFrame:
    """
    Pulls per-player boxscore stats for a single game.
    IMPORTANT: BallDontLie v1 uses `turnover` (singular), not `turnovers`.
    """
    rows = []
    cursor = 0

    while True:
        data = safe_get(
            BASE_URL + "stats",
            headers=HEADERS,
            params={"game_ids[]": game_id, "per_page": 100, "cursor": cursor},
        )

        items = data.get("data", [])
        if not items:
            break

        for st in items:
            player = st.get("player") or {}
            team = st.get("team") or {}

            # ✅ FIX: use turnover (singular), fallback to turnovers just in case
            to_val = st.get("turnover")
            if to_val is None:
                to_val = st.get("turnovers")

            rows.append({
                "player_name": f"{player.get('first_name','')} {player.get('last_name','')}".strip(),
                "team": team.get("abbreviation"),
                "pts": st.get("pts"),
                "reb": st.get("reb"),
                "ast": st.get("ast"),
                "stl": st.get("stl"),
                "blk": st.get("blk"),
                "to": to_val,
            })

        meta = data.get("meta") or {}
        if meta.get("next_cursor") is None:
            break
        cursor = meta["next_cursor"]
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Fill missing with 0 before aggregation
    for c in ["pts", "reb", "ast", "stl", "blk", "to"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def build_actuals_for_date(date_str: str) -> pd.DataFrame:
    games = get_games_for_date(date_str)
    if not games:
        print(f"⚠️ No games found for {date_str}")
        return pd.DataFrame()

    all_stats = []
    for g in games:
        df_g = get_stats_for_game(g["id"])
        if not df_g.empty:
            all_stats.append(df_g)

    if not all_stats:
        return pd.DataFrame()

    df = pd.concat(all_stats, ignore_index=True)

    df["player_key"] = df["player_name"].apply(normalize_name)

    # Aggregate (some endpoints can return multiple rows per player)
    df = (
        df.groupby(["player_key", "player_name"], as_index=False)[["pts","reb","ast","stl","blk","to"]]
          .sum()
    )

    # Combos
    df["pr"] = df["pts"] + df["reb"]
    df["pa"] = df["pts"] + df["ast"]
    df["ra"] = df["reb"] + df["ast"]
    df["pra"] = df["pts"] + df["reb"] + df["ast"]

    # Fantasy
    df["fantasy"] = (
        df["pts"]
        + 1.2 * df["reb"]
        + 1.5 * df["ast"]
        + 3.0 * df["stl"]
        + 3.0 * df["blk"]
    )

    return df


# -----------------------
# GRADING
# -----------------------
def compute_actual_value(row, actuals_row):
    prop = str(row.get("prop")).upper().strip()

    mapping = {
        "PTS": "pts",
        "REB": "reb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TO": "to",
        "PR": "pr",
        "PA": "pa",
        "RA": "ra",
        "PRA": "pra",
        "FANTASY": "fantasy",
    }

    col = mapping.get(prop)
    if not col:
        return np.nan
    return float(actuals_row[col])


def grade_pick(direction: str, actual: float, line: float):
    if pd.isna(actual) or pd.isna(line):
        return "NO_DATA"

    if abs(actual - line) < 1e-9:
        return "PUSH"

    direction = str(direction).upper().strip()

    if direction == "OVER":
        return "WIN" if actual > line else "LOSS"
    if direction == "UNDER":
        return "WIN" if actual < line else "LOSS"

    return "UNKNOWN"


# -----------------------
# MAIN
# -----------------------
def main(target_date: str = None):
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    board_path = BOARDS_DIR / f"board_{target_date}.csv"
    if not board_path.exists():
        raise FileNotFoundError(f"Missing {board_path}")

    board = pd.read_csv(board_path)

    board["player_key"] = board["player"].apply(normalize_name)
    board["book_line"] = pd.to_numeric(board["book_line"], errors="coerce")

    actuals = build_actuals_for_date(target_date)
    if actuals.empty:
        print(f"⚠️ No actual stats available for {target_date}")
        return

    settled = board.merge(actuals, on="player_key", how="left")

    settled["actual_value"] = settled.apply(lambda r: compute_actual_value(r, r), axis=1)
    settled["result"] = settled.apply(
        lambda r: grade_pick(r["direction"], r["actual_value"], r["book_line"]),
        axis=1
    )

    keep_cols = [
        "player", "prop", "book_line", "direction",
        "model_prediction", "edge", "actual_value", "result"
    ]

    settled_out = settled.loc[:, keep_cols].copy()

    out_path = SETTLED_DIR / f"settled_{target_date}.csv"
    settled_out.to_csv(out_path, index=False)

    print(f"\n✅ Settled results saved -> {out_path}")
    print(settled_out["result"].value_counts(dropna=False).to_string())

    # Optional quick sanity check for TO rows
    to_rows = settled_out[settled_out["prop"].astype(str).str.upper().str.strip() == "TO"]
    if not to_rows.empty:
        zeros = (to_rows["actual_value"] == 0).sum()
        print(f"\n🔎 TO sanity check: {len(to_rows)} TO rows, {zeros} have actual_value == 0")


if __name__ == "__main__":
    main()