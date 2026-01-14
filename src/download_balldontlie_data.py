from __future__ import annotations
import os
import math
import time
import requests
import pandas as pd
from dotenv import load_dotenv

from config import RAW_DIR, RAW_BOXSCORES_CSV

load_dotenv()

API_KEY = os.getenv("BALLDONTLIE_API_KEY")

BASE_URL = "https://api.balldontlie.io/v1/stats"


HEADERS = {
    "Authorization": API_KEY
}


def fetch_stats(seasons: list[int]) -> pd.DataFrame:
    all_rows = []

    for season in seasons:
        print(f"\nPulling season: {season}")

        page = 1
        per_page = 100

        while True:
            url = f"{BASE_URL}?seasons[]={season}&per_page={per_page}&page={page}"
            r = requests.get(url, headers=HEADERS)

            if r.status_code != 200:
                print("Error:", r.text)
                break

            data = r.json()
            stats = data.get("data", [])

            if not stats:
                break

            for s in stats:
                player = s["player"]
                team = s["team"]
                game = s["game"]
                stat = s["stats"]

                all_rows.append({
                    "date": game["date"].split("T")[0],
                    "player": f"{player['first_name']} {player['last_name']}",
                    "team": team["abbreviation"],
                    "opponent": game["home_team_id"] if team["id"] != game["home_team_id"] else game["visitor_team_id"],
                    "min": stat["min"] if stat["min"] else 0,
                    "pts": stat["pts"],
                    "reb": stat["reb"],
                    "ast": stat["ast"],
                })

            print(f"Pulled page {page} ({len(all_rows)} total rows)")
            page += 1

            total_pages = data["meta"]["total_pages"]
            if page > total_pages:
                break

            time.sleep(0.4)

    df = pd.DataFrame(all_rows)
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    seasons = [2022, 2023, 2024]

    df = fetch_stats(seasons)

    print(f"\nFinal Rows: {len(df)}")

    df.to_csv(RAW_BOXSCORES_CSV, index=False)
    print(f"Saved historical dataset to: {RAW_BOXSCORES_CSV}")


if __name__ == "__main__":
    main()
