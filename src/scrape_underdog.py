import json
import requests
from pathlib import Path

OUT = Path("data/props/underdog.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = "https://api.underdogfantasy.com/beta/v3/over_under_lines"


def scrape_underdog():
    print("Fetching Underdog props...")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    res = requests.get(URL, headers=headers, timeout=15)

    if res.status_code != 200:
        raise RuntimeError(f"Underdog HTTP {res.status_code}")

    data = res.json()

    lines = data.get("over_under_lines", [])
    markets = {m["id"]: m for m in data.get("over_unders", [])}
    players = {p["id"]: p for p in data.get("players", [])}

    print(f"Underdog returned {len(lines)} total lines")

    if len(lines) == 0:
        print("⚠️ Underdog returned NO props at all. Saving empty list.")
        OUT.write_text("[]")
        return

    props = []
    skipped_not_nba = 0

    for line in lines:
        ou = markets.get(line["over_under_id"])
        if not ou:
            continue

        pid = str(ou["player_id"])
        player = players.get(pid)
        if not player:
            continue

        # League detection (UD sometimes changes fields)
        league = (
            player.get("sport_slug")
            or player.get("competition")
            or player.get("league")
            or ""
        ).lower()

        # Only NBA
        if "nba" not in league and "basketball" not in league:
            skipped_not_nba += 1
            continue

        props.append({
            "player": player["name"],
            "team": player.get("team_name"),
            "stat": ou["appearance_stat_type"].upper(),
            "line": float(line["stat_value"]),
            "source": "Underdog"
        })

    with open(OUT, "w") as f:
        json.dump(props, f, indent=2)

    print(f"Skipped NON-NBA: {skipped_not_nba}")
    print(f"Saved {len(props)} NBA Underdog props → {OUT}")

    # 🔥 Optional Improvement Message
    if len(props) == 0:
        print("⚠️ No NBA props currently available on Underdog.")
        print("👉 This is NORMAL if NBA board hasn't posted yet. Try again later.")


if __name__ == "__main__":
    scrape_underdog()









