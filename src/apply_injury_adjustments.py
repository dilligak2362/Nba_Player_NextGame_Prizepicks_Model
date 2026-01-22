import pandas as pd
from pathlib import Path

# ===========================
# PATHS
# ===========================
DATASET_PATH = Path("data/processed/inference_dataset.csv")
INJURIES_PATH = Path("data/injuries/injuries_latest.csv")
OUT_PATH = Path("data/processed/inference_dataset_adjusted.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===========================
# CONFIG
# ===========================
IMPACT_STATUSES = {"OUT", "DOUBTFUL"}

USAGE_BUMP = 0.10   # +10%
MIN_BUMP = 0.06     # +6%

USAGE_COL_CANDIDATES = ["roll_usage", "ewm_usage", "usage", "usg", "usg_rate"]
MIN_COL_CANDIDATES = ["roll_min", "ewm_min", "min", "minutes", "proj_min_proxy"]


def main():
    print("📊 Loading inference dataset...")
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.lower().strip() for c in df.columns]

    if "player_name" in df.columns and "player" not in df.columns:
        df.rename(columns={"player_name": "player"}, inplace=True)

    if "team" not in df.columns:
        print("❌ Dataset missing 'team' column. Saving passthrough.")
        df.to_csv(OUT_PATH, index=False)
        return

    if not INJURIES_PATH.exists():
        print("⚠️ Injury file missing. Passthrough mode.")
        df.to_csv(OUT_PATH, index=False)
        return

    inj = pd.read_csv(INJURIES_PATH)
    inj.columns = [c.lower().strip() for c in inj.columns]

    if inj.empty:
        print("⚠️ Injury file has 0 rows. Passthrough mode.")
        df.to_csv(OUT_PATH, index=False)
        return

    required = {"team", "status"}
    if not required.issubset(inj.columns):
        print("⚠️ Injury file missing required columns. Passthrough mode.")
        df.to_csv(OUT_PATH, index=False)
        return

    inj["team"] = inj["team"].astype(str).str.strip()
    inj["status"] = inj["status"].astype(str).str.upper().str.strip()

    impacted_teams = inj[inj["status"].isin(IMPACT_STATUSES)]["team"].dropna().unique().tolist()

    if not impacted_teams:
        print("ℹ️ No OUT/DOUBTFUL injuries today. Passthrough mode.")
        df.to_csv(OUT_PATH, index=False)
        return

    print(f"🩺 Impacted teams: {impacted_teams}")

    df["injury_boost"] = df["team"].astype(str).str.strip().isin(impacted_teams).astype(int)

    def bump_cols(col_list, bump, label):
        applied = []
        for c in col_list:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                df.loc[df["injury_boost"] == 1, c] *= (1.0 + bump)
                applied.append(c)

        if applied:
            print(f"✅ Applied {label} bump to {applied}")
        else:
            print(f"⚠️ No {label} columns found")

    bump_cols(USAGE_COL_CANDIDATES, USAGE_BUMP, "USAGE")
    bump_cols(MIN_COL_CANDIDATES, MIN_BUMP, "MINUTES")

    df.to_csv(OUT_PATH, index=False)

    print(f"✅ Saved injury-adjusted dataset → {OUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Boosted rows: {int(df['injury_boost'].sum())}")


if __name__ == "__main__":
    main()