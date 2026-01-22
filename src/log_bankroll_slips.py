import pandas as pd
from pathlib import Path
from datetime import datetime

SLIPS_DIR = Path("data/bankroll_builder/daily_slips")
BET_LOG = Path("data/bankroll/bet_log.csv")
BET_LOG.parent.mkdir(parents=True, exist_ok=True)

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    slips_path = SLIPS_DIR / f"builder_slips_{today}.csv"

    if not slips_path.exists():
        raise FileNotFoundError(f"No slips found for {today}")

    slips = pd.read_csv(slips_path)

    rows = []
    for _, r in slips.iterrows():
        rows.append({
            "date": today,
            "slip_id": r["slip_id"],
            "players": r["players"],
            "props": r["props"],
            "directions": r["directions"],
            "stake": r["slip_size"],
            "expected_value": r["expected_value"],
            "result": "PENDING",
            "pnl": 0.0
        })

    new_log = pd.DataFrame(rows)

    if BET_LOG.exists():
        log = pd.read_csv(BET_LOG)
        log = pd.concat([log, new_log], ignore_index=True)
    else:
        log = new_log

    log.to_csv(BET_LOG, index=False)
    print(f"✅ Logged {len(new_log)} bets → {BET_LOG}")

if __name__ == "__main__":
    main()