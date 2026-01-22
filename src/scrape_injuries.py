import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# ===========================
# CONFIG
# ===========================
OUT_DIR = Path("data/injuries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LATEST = OUT_DIR / "injuries_latest.csv"
OUT_DATED = OUT_DIR / f"injuries_{datetime.now().strftime('%Y-%m-%d')}.csv"

UNDERDOG_URL = "https://api.underdogfantasy.com/v2/sports/basketball/nba/injuries"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

# ===========================
# STATUS NORMALIZATION
# ===========================
def normalize_status(s):
    if not s:
        return ""
    s = s.upper()
    if "OUT" in s:
        return "OUT"
    if "DOUBT" in s:
        return "DOUBTFUL"
    if "QUESTION" in s:
        return "QUESTIONABLE"
    if "PROB" in s:
        return "PROBABLE"
    return s


# ===========================
# MAIN
# ===========================
def main():
    print("🩺 Fetching Underdog NBA injuries...")

    try:
        r = requests.get(UNDERDOG_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"❌ Injury feed failed: {e}")
        empty = pd.DataFrame(columns=["player", "team", "status", "description", "date"])
        empty.to_csv(OUT_LATEST, index=False)
        empty.to_csv(OUT_DATED, index=False)
        print("Saved empty injury file (passthrough mode)")
        return

    rows = []

    for item in data.get("injuries", []):
        player = item.get("player_name")
        team = item.get("team_abbreviation")
        status = normalize_status(item.get("status"))
        desc = item.get("injury") or ""
        date = item.get("updated_at") or ""

        if player and team:
            rows.append({
                "player": player.strip(),
                "team": team.strip(),
                "status": status,
                "description": desc.strip(),
                "date": date
            })

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=["player", "team", "status", "description", "date"])

    df.to_csv(OUT_LATEST, index=False)
    df.to_csv(OUT_DATED, index=False)

    print(f"✅ Saved injuries → {OUT_LATEST}")
    print(f"Rows: {len(df)}")

    if len(df) > 0:
        print(df.head(10))


if __name__ == "__main__":
    main()