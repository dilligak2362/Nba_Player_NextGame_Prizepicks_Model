import pandas as pd
import numpy as np

PREDICTIONS_PATH = "data/processed/model_predictions.csv"

# Expected drivers
DRIVERS = [
    "proj_min",
    "roll_min",
    "ewm_min",
    "usage",
    "roll_usage",
    "ewm_usage",
    "usg_rate",
    "pace_proxy",
    "opp_pace",
    "opp_def_rating"
]

TARGET = "model_prediction"


def main():
    print("\n==============================")
    print("🔎 MODEL INPUT VALIDATION")
    print("==============================\n")

    df = pd.read_csv(PREDICTIONS_PATH)
    df.columns = [c.lower().strip() for c in df.columns]

    if TARGET not in df.columns:
        print(f"❌ {TARGET} column not found in predictions file")
        print("Available columns:\n")
        print(df.columns.tolist())
        return

    print(f"Loaded predictions file: {len(df)} rows\n")

    # ===========================
    # 1. Column existence check
    # ===========================
    print("▶ Checking expected input columns:\n")

    found = []
    missing = []

    for col in DRIVERS:
        if col in df.columns:
            found.append(col)
            print(f"✅ {col}")
        else:
            missing.append(col)
            print(f"❌ {col} (MISSING)")

    # ===========================
    # 2. Variance check
    # ===========================
    print("\n▶ Checking for dead / constant columns:\n")

    dead = []

    for col in found:
        std = df[col].std()
        if std < 1e-6:
            print(f"⚠️ {col} is constant (std ≈ 0)")
            dead.append(col)
        else:
            print(f"✅ {col} variance OK (std = {std:.4f})")

    # ===========================
    # 3. Correlation with projection
    # ===========================
    print("\n▶ Correlation with model projection:\n")

    correlations = {}

    for col in found:
        corr = df[col].corr(df[TARGET])
        correlations[col] = corr
        print(f"📈 Corr({col}, projection) = {corr:.4f}")

    # ===========================
    # 4. Rank importance
    # ===========================
    print("\n▶ Input importance ranking:\n")

    ranked = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for i, (col, corr) in enumerate(ranked, 1):
        print(f"{i:>2}. {col:<20} | corr = {corr:.4f}")

    # ===========================
    # 5. Projection sanity
    # ===========================
    print("\n▶ Projection distribution:\n")
    print(df[TARGET].describe())

    # ===========================
    # Final verdict
    # ===========================
    print("\n==============================")
    print("📋 MODEL INPUT VERDICT")
    print("==============================\n")

    if missing:
        print(f"❌ Missing drivers: {missing}")
    else:
        print("✅ All expected drivers present")

    if dead:
        print(f"⚠️ Dead inputs (no variance): {dead}")
    else:
        print("✅ No dead inputs")

    if ranked:
        print(f"\n🏆 Strongest driver: {ranked[0][0]}")

    print("\n✅ Model input validation complete.\n")


if __name__ == "__main__":
    main()