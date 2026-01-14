import pandas as pd
from pathlib import Path
from datetime import datetime

BOARD_IN = Path("data/processed/merged_props_predictions.csv")

OUT_DIR = Path("data/history/boards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not BOARD_IN.exists():
        raise FileNotFoundError(f"Missing {BOARD_IN}. Run merge_props_with_predictions.py first.")

    df = pd.read_csv(BOARD_IN)

    # Add run metadata
    run_date = datetime.now().strftime("%Y-%m-%d")
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["run_date"] = run_date
    df["run_timestamp"] = run_ts

    out_path = OUT_DIR / f"board_{run_date}.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved daily board snapshot -> {out_path}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()