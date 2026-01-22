import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("data/injuries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://www.fantasylabs.com/api/players/injuries"


def scrape_fantasylabs():
    print("🏥 Fetching FantasyLabs injury feed...")

    r = requests.get(URL, timeout=20)
    r.raise_for_status()

    data = r.json()

    rows = []

    for p in data:
        rows.append({
            "player": p.get("name", "").strip(),
            "team": p.get("team", "").strip(),
            "status": p.get("status", "").upper().strip(),
            "injury": p.get("injury", "").strip(),
            "source": "FantasyLabs",
            "updated": p.get("updated", "")
        })

    df = pd.DataFrame(rows)

    today = datetime.now().strftime("%Y-%m-%d")

    latest_path = OUT_DIR / "injuries_latest.csv"
    dated_path = OUT_DIR / f"injuries_{today}.csv"

    df.to_csv(latest_path, index=False)
    df.to_csv(dated_path, index=False)

    print(f"✅ Saved injuries → {latest_path}")
    print(f"Rows: {len(df)}")

    # Show sample
    print("\nSample:")
    print(df.head(10))


if __name__ == "__main__":
    scrape_fantasylabs()