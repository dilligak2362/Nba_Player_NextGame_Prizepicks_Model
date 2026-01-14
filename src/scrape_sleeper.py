import requests
import json
import time
from pathlib import Path

OUT = Path("data/props/sleeper.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

SLEEPER_URL = "https://api.sleeper.app/v1/picks/nba"


def scrape_sleeper(max_retries=6):
    print("Fetching Sleeper props...")

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(SLEEPER_URL, timeout=15)

            # Sleeper frequently throws 500 when board is closed
            if r.status_code == 500:
                wait = attempt * 3
                print(f"⚠️ Sleeper API 500 (attempt {attempt}/{max_retries}) — retrying in {wait}s")
                time.sleep(wait)
                continue

            # other bad status
            if r.status_code != 200:
                print(f"⚠️ Sleeper API returned {r.status_code}")
                time.sleep(2)
                continue

            data = r.json()

            props = []

            if isinstance(data, list):
                for p in data:
                    player = None
                    if isinstance(p.get("player"), dict):
                        player = p["player"].get("full_name")
                    else:
                        player = p.get("player")

                    stat = p.get("stat")
                    line = p.get("line")
                    team = p.get("team")

                    if not player or not stat or line is None:
                        continue

                    try:
                        line = float(line)
                    except:
                        continue

                    props.append({
                        "player": str(player).strip(),
                        "team": team,
                        "stat": str(stat).strip(),
                        "line": line,
                        "prop_type": "single",
                        "source": "Sleeper",
                    })

            # Always save file
            with open(OUT, "w") as f:
                json.dump(props, f, indent=2)

            print(f"✅ Saved Sleeper props → {OUT} ({len(props)} props)")
            return

        except Exception as e:
            wait = attempt * 3
            print(f"⚠️ Sleeper error (attempt {attempt}/{max_retries}): {e}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    # If all retries fail — save empty file and move on
    print("⚠️ Sleeper unavailable. Saving empty board.")
    with open(OUT, "w") as f:
        json.dump([], f, indent=2)


if __name__ == "__main__":
    scrape_sleeper()