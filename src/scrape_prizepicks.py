import json
import time
import requests
from pathlib import Path

OUT = Path("data/props/prizepicks.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# NBA = 7 on PrizePicks
URL = "https://api.prizepicks.com/projections?per_page=1000&league_id=7"


def scrape_prizepicks():
    print("Fetching PrizePicks props...")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Origin": "https://app.prizepicks.com",
        "Referer": "https://app.prizepicks.com/",
    }

    # -------- safe request --------
    data = None
    for attempt in range(4):
        try:
            res = requests.get(URL, headers=headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                break
            else:
                print(f"HTTP {res.status_code}, retrying…")
        except Exception as e:
            print(f"HTTP error attempt {attempt+1}: {e}")
        time.sleep(2 + attempt)

    if data is None:
        raise RuntimeError("PrizePicks failed after retries")

    projections = data.get("data", [])
    included = data.get("included", [])

    # =========================
    # Player lookup
    # =========================
    players = {}
    for item in included:
        if item.get("type") != "new_player":
            continue

        attrs = item.get("attributes", {}) or {}
        if "NBA" not in (attrs.get("league") or "").upper():
            continue

        players[item["id"]] = {
            "name": attrs.get("display_name"),
            "team": attrs.get("team_name"),
        }

    print(f"Found {len(players)} NBA players")

    # =========================
    # Helpers
    # =========================
    def normalize_stat(raw):
        """
        Robust stat normalizer for PrizePicks.
        Handles:
          - Singles: PTS, REB, AST, STL, BLK, TO
          - Combos: PR, PA, RA, PRA
        """
        if not raw:
            return None

        s = str(raw).upper().strip()

        # Normalize separators
        s = s.replace(" ", "")
        s = s.replace("/", "+")
        s = s.replace("&", "+")
        s = s.replace("-", "")

        # Normalize words → abbreviations
        s = (
            s.replace("POINTS", "PTS")
             .replace("PTS", "PTS")
             .replace("REBOUNDS", "REB")
             .replace("REBS", "REB")
             .replace("REB", "REB")
             .replace("ASSISTS", "AST")
             .replace("ASTS", "AST")
             .replace("AST", "AST")
             .replace("STEALS", "STL")
             .replace("STL", "STL")
             .replace("BLOCKS", "BLK")
             .replace("BLK", "BLK")
             .replace("TURNOVERS", "TO")
             .replace("TURNOVER", "TO")
             .replace("TOV", "TO")
             .replace("TO", "TO")
        )

        # Singles
        if s in {"PTS", "REB", "AST", "STL", "BLK", "TO"}:
            return s

        # Combos (check PRA first)
        if "+" in s:
            parts = {p for p in s.split("+") if p}

            if {"PTS", "REB", "AST"}.issubset(parts):
                return "PRA"
            if {"PTS", "REB"}.issubset(parts):
                return "PR"
            if {"PTS", "AST"}.issubset(parts):
                return "PA"
            if {"REB", "AST"}.issubset(parts):
                return "RA"

        return None

    def is_full_game(attr):
        text = (
            (attr.get("description") or "")
            + " "
            + (attr.get("title") or "")
            + " "
            + (attr.get("market_type") or "")
        ).lower()

        bad = [
            "1q", "2q", "3q", "4q",
            "quarter", "half",
            "1sthalf", "2ndhalf",
            "1st half", "2nd half",
        ]
        return not any(b in text for b in bad)

    # =========================
    # Extract props
    # =========================
    props = []

    for p in projections:
        attr = p.get("attributes", {}) or {}

        # ✅ Only STANDARD (no goblins/demons)
        if attr.get("odds_type") != "standard":
            continue

        if not is_full_game(attr):
            continue

        raw_stat = (
            attr.get("stat_type")
            or attr.get("market_type")
            or attr.get("title")
            or attr.get("description")
        )

        stat = normalize_stat(raw_stat)
        if not stat:
            continue

        # Resolve player
        pid = attr.get("player_id")
        if not pid:
            rel = p.get("relationships", {}).get("new_player", {}).get("data")
            pid = rel.get("id") if isinstance(rel, dict) else None

        if not pid or pid not in players:
            continue

        try:
            line = float(attr.get("line_score"))
        except Exception:
            continue

        prop_type = "combo" if stat in {"PR", "PA", "RA", "PRA"} else "standard"

        props.append({
            "player": players[pid]["name"],
            "team": players[pid]["team"],
            "stat": stat,
            "line": line,
            "source": "PrizePicks",
            "prop_type": prop_type,
        })

    with open(OUT, "w") as f:
        json.dump(props, f, indent=2)

    print(f"✅ Saved {len(props)} PrizePicks props (standard + combo + TO) → {OUT}")

    # Sanity check
    try:
        import pandas as pd
        df = pd.DataFrame(props)
        print("\nStat counts:")
        print(df["stat"].value_counts())
        print("\nProp type counts:")
        print(df["prop_type"].value_counts())
    except Exception:
        pass

    


if __name__ == "__main__":
    scrape_prizepicks()