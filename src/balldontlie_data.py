import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================ CONFIG ============================

BASE_URL = "https://api.balldontlie.io/v1"

API_KEY = os.getenv("BALLDONTLIE_API_KEY") or "6d8598ea-cdfa-4e30-a830-9e758a22b66d"

REQUEST_DELAY = 0.25
MAX_GAMES_PER_SEASON = 6000

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RAW_DIR / "historical_boxscores.csv"

CHECKPOINT_EVERY = 1000


# ============================ HTTP HELPERS ============================

def safe_get(endpoint: str, params=None, max_retries: int = 8, backoff_base: float = 2.0):
    if params is None:
        params = {}

    url = f"{BASE_URL}/{endpoint}"
    headers = {"Authorization": API_KEY}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)

            if resp.status_code == 429:
                wait = backoff_base * (attempt + 1)
                print(f"⚠️ Rate limit (429) — retry {attempt+1}/{max_retries} in {wait:.1f}s…")
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                raise RuntimeError("❌ 401 Unauthorized. Check API key.")

            if not resp.ok:
                if 500 <= resp.status_code < 600:
                    wait = backoff_base * (attempt + 1)
                    print(f"Server error {resp.status_code} — retrying in {wait:.1f}s…")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

            return resp.json()

        except (requests.ConnectionError, requests.Timeout) as e:
            wait = backoff_base * (attempt + 1)
            print(f"🌐 Network issue: {repr(e)} — retry {attempt+1}/{max_retries} in {wait:.1f}s…")
            time.sleep(wait)

    raise RuntimeError(f"safe_get FAILED for {url}")


# ============================ TEAMS ============================

def load_team_lookup():
    print("\nLoading NBA teams…")
    lookup = {}
    cursor = 0
    page = 1

    while True:
        data = safe_get("teams", params={"per_page": 100, "cursor": cursor})
        items = data.get("data", [])
        if not items:
            break

        for t in items:
            lookup[t["id"]] = t["abbreviation"]

        meta = data.get("meta") or {}
        next_cursor = meta.get("next_cursor")

        print(f"Teams page {page}: {len(items)} teams")

        if next_cursor is None:
            break

        cursor = next_cursor
        page += 1
        time.sleep(REQUEST_DELAY)

    print(f"Loaded {len(lookup)} NBA teams.\n")
    return lookup


# ============================ GAMES ============================

def fetch_games_for_season(season: int, max_games: int = MAX_GAMES_PER_SEASON):
    print(f"\n===== FETCHING GAMES FOR {season} =====")

    games = []
    cursor = 0
    page = 1

    while True:
        data = safe_get(
            "games",
            params={
                "seasons[]": season,
                "per_page": 100,
                "cursor": cursor,
                "postseason": "false",
            }
        )

        items = data.get("data", [])
        if not items:
            break

        games.extend(items)

        meta = data.get("meta") or {}
        next_cursor = meta.get("next_cursor")

        print(f"Page {page} — got {len(items)} games (total: {len(games)})")

        if len(games) >= max_games:
            print(f"🛑 Safety stop at {max_games} games")
            break

        if next_cursor is None:
            break

        cursor = next_cursor
        page += 1
        time.sleep(REQUEST_DELAY)

    print(f"✅ Season {season} total games: {len(games)}\n")
    return games


# ============================ BOXSCORES ============================

def fetch_boxscores(all_games, team_lookup):
    rows = []
    total_games = len(all_games)

    print(f"Fetching boxscores for {total_games} games…")

    for idx, g in enumerate(all_games, start=1):
        gid = g["id"]
        season = g.get("season")

        cursor = 0

        while True:
            data = safe_get(
                "stats",
                params={
                    "game_ids[]": gid,
                    "per_page": 100,
                    "cursor": cursor,
                }
            )

            stats_list = data.get("data", [])
            if not stats_list:
                break

            for stat in stats_list:
                game_info = stat["game"]
                player = stat["player"]
                team_id = stat["team"]["id"]

                home_id = game_info["home_team_id"]
                visitor_id = game_info["visitor_team_id"]
                opp_id = visitor_id if team_id == home_id else home_id

                rows.append({
                    "season": season,
                    "game_id": gid,
                    "date": game_info["date"].split("T")[0],
                    "player_id": player["id"],
                    "player_name": f"{player['first_name']} {player['last_name']}",
                    "team": team_lookup.get(team_id, "UNK"),
                    "opponent": team_lookup.get(opp_id, "UNK"),

                    # Core stats
                    "pts": stat.get("pts", 0),
                    "reb": stat.get("reb", 0),
                    "ast": stat.get("ast", 0),
                    "stl": stat.get("stl", 0),
                    "blk": stat.get("blk", 0),

                    # ✅ Turnovers (FIXED FIELD)
                    "to": stat.get("turnovers", 0),

                    # Possession context
                    "fga": stat.get("fga", 0),
                    "fta": stat.get("fta", 0),
                    "oreb": stat.get("oreb", 0),

                    # Minutes
                    "min": stat.get("min", 0),
                })

            meta = data.get("meta") or {}
            next_cursor = meta.get("next_cursor")
            if next_cursor is None:
                break

            cursor = next_cursor
            time.sleep(REQUEST_DELAY)

        if idx % 250 == 0 or idx == total_games:
            print(f"Processed {idx}/{total_games} games…")

        if idx % CHECKPOINT_EVERY == 0 or idx == total_games:
            tmp_df = pd.DataFrame(rows)
            tmp_path = RAW_DIR / "historical_boxscores_checkpoint.csv"
            tmp_df.to_csv(tmp_path, index=False)
            print(f"💾 Checkpoint saved -> {tmp_path} ({len(tmp_df)} rows)")

    df = pd.DataFrame(rows)
    print(f"\nFINAL ROW COUNT: {len(df)}")
    return df


# ============================ MAIN ============================

def main():
    print("===== NBA BallDontLie Data Collection =====")
    print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")

    if not API_KEY:
        raise RuntimeError("❌ No valid API key set.")

    team_lookup = load_team_lookup()

    seasons = [2024, 2025]

    all_games = []
    for s in seasons:
        games = fetch_games_for_season(s)
        for g in games:
            if "season" not in g:
                g["season"] = s
        all_games.extend(games)

    print(f"Total games across seasons: {len(all_games)}")

    df = fetch_boxscores(all_games, team_lookup)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved full dataset -> {OUTPUT_FILE}")
    print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
