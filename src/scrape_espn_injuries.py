# src/scrape_injuries.py
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests

OUT_DIR = Path("data/injuries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESPN_URL = "https://www.espn.com/nba/injuries"

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

def _normalize_player(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name

def _normalize_status(s: str) -> str:
    if pd.isna(s):
        return "UNKNOWN"
    s = str(s).strip().upper()
    # ESPN usually: "Out", "Questionable", "Doubtful", "Day-To-Day", etc.
    if "OUT" in s:
        return "OUT"
    if "DOUBT" in s:
        return "DOUBTFUL"
    if "QUESTION" in s:
        return "QUESTIONABLE"
    if "DAY" in s:
        return "DAY_TO_DAY"
    if "PROB" in s:
        return "PROBABLE"
    return s.replace(" ", "_")

def main():
    print("📡 Fetching ESPN injuries page...")
    r = requests.get(
        ESPN_URL,
        timeout=25,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
        },
    )
    r.raise_for_status()
    html = r.text

    # Parse all tables
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("❌ ESPN returned no injury tables.")

    # Try to get team names from headings (best-effort)
    team_names = re.findall(r'(<h2[^>]*>)([^<]+)(</h2>)', html)
    team_names = [t[1].strip() for t in team_names if t[1].strip() in TEAM_NAME_TO_ABBR]

    rows = []
    team_idx = 0

    for t in tables:
        # ESPN tables commonly include columns like: PLAYER, POS, DATE, INJURY, STATUS
        cols = [str(c).strip().upper() for c in t.columns]
        t.columns = cols

        if "PLAYER" not in t.columns:
            continue
        if "STATUS" not in t.columns:
            continue

        # Assign team if possible
        team_name = None
        if "TEAM" in t.columns:
            # sometimes a team column exists
            team_name = None
        else:
            # use heading list alignment if we have it
            if team_idx < len(team_names):
                team_name = team_names[team_idx]
                team_idx += 1

        for _, rrow in t.iterrows():
            player = _normalize_player(rrow.get("PLAYER", ""))
            status_raw = rrow.get("STATUS", "")
            status = _normalize_status(status_raw)

            injury = rrow.get("INJURY", "")
            date = rrow.get("DATE", "")

            team_abbr = None
            if "TEAM" in t.columns:
                tn = str(rrow.get("TEAM", "")).strip()
                team_abbr = TEAM_NAME_TO_ABBR.get(tn) or TEAM_NAME_TO_ABBR.get(tn.title())
            elif team_name:
                team_abbr = TEAM_NAME_TO_ABBR.get(team_name)

            if not player:
                continue

            rows.append(
                {
                    "team": team_abbr if team_abbr else "",
                    "team_name": team_name if team_name else "",
                    "player": player,
                    "status": status,
                    "status_raw": str(status_raw),
                    "injury": "" if pd.isna(injury) else str(injury),
                    "date": "" if pd.isna(date) else str(date),
                    "source": "ESPN",
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset=["team", "player", "status", "injury"], keep="first")

    today = datetime.now().strftime("%Y-%m-%d")
    latest_path = OUT_DIR / "injuries_latest.csv"
    dated_path = OUT_DIR / f"injuries_{today}.csv"

    df.to_csv(latest_path, index=False)
    df.to_csv(dated_path, index=False)

    print(f"✅ Saved injuries → {latest_path}")
    print(f"✅ Saved injuries → {dated_path}")
    print(f"Rows: {len(df)}")
    if len(df) > 0:
        print(df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()