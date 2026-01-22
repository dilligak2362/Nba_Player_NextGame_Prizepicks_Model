import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PARLAY_SLIPS_PATH = Path("data/calibration_parlays/calibration_parlay_slips.csv")
CALIB_RESULTS_DIR = Path("data/calibration_results")
OUT_DIR = Path("data/calibration_parlays")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(target_date=None):
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    results_path = CALIB_RESULTS_DIR / f"calibration_results_{target_date}.csv"

    if not PARLAY_SLIPS_PATH.exists():
        raise FileNotFoundError(f"Missing {PARLAY_SLIPS_PATH}")

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    print("\n📥 Loading calibration parlay slips...")
    parlays = pd.read_csv(PARLAY_SLIPS_PATH)

    print("📥 Loading calibration settled props...")
    settled = pd.read_csv(results_path)

    settled["key"] = settled["player"] + "|" + settled["prop"]
    result_map = dict(zip(settled["key"], settled["result"]))

    parlay_rows = []

    for _, row in parlays.iterrows():
        players = row["players"].split(" | ")
        props = row["props"].split(" | ")

        leg_results = []
        for p, pr in zip(players, props):
            key = f"{p}|{pr}"
            leg_results.append(result_map.get(key, "MISSING"))

        if "LOSS" in leg_results:
            final = "LOSS"
        elif "PUSH" in leg_results:
            final = "PUSH"
        elif "MISSING" in leg_results:
            final = "MISSING"
        else:
            final = "WIN"

        parlay_rows.append({
            **row,
            "result": final,
            "legs_results": " | ".join(leg_results)
        })

    out = pd.DataFrame(parlay_rows)

    out_path = OUT_DIR / f"calibration_settled_parlays_{target_date}.csv"
    out.to_csv(out_path, index=False)

    print(f"\n✅ Calibration parlay settlement saved → {out_path}")
    print(out["result"].value_counts(dropna=False))


if __name__ == "__main__":
    main()