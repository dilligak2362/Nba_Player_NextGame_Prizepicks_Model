import pandas as pd
from pathlib import Path
from joblib import load

BOARD = Path("data/processed/merged_props_predictions.csv")
MODEL = Path("models/calibration_model.joblib")
OUT = Path("data/processed/calibrated_board.csv")

OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("\n📥 Loading merged board...")
    df = pd.read_csv(BOARD)

    print("🤖 Loading calibration model...")
    model = load(MODEL)

    # -------------------------
    # Normalize stat column
    # -------------------------
    df["stat"] = df["prop"].astype(str).str.upper().str.strip()

    # -------------------------
    # Ensure proj_min exists
    # -------------------------
    if "proj_min" not in df.columns:
        print("⚠️ proj_min missing from board — filling with 0")
        df["proj_min"] = 0.0

    # -------------------------
    # Drop bad rows
    # -------------------------
    df = df.dropna(subset=["model_prediction", "book_line", "edge"])

    # -------------------------
    # Build calibration feature frame
    # -------------------------
    cal = df[["model_prediction", "book_line", "proj_min", "edge", "stat"]].copy()

    # One-hot encode stat
    cal = pd.get_dummies(cal, columns=["stat"], prefix="stat")

    # -------------------------
    # Align columns to trained model
    # -------------------------
    for c in model.feature_names_in_:
        if c not in cal.columns:
            cal[c] = 0.0

    cal = cal[model.feature_names_in_]

    # -------------------------
    # Convert all to float
    # -------------------------
    cal = cal.astype(float)

    # -------------------------
    # Predict calibrated projection
    # -------------------------
    df["true_projection"] = model.predict(cal)
    df["true_edge"] = df["true_projection"] - df["book_line"]

    # -------------------------
    # SORT: Highest → Lowest Edge
    # -------------------------
    df = df.sort_values("true_edge", ascending=False).reset_index(drop=True)

    # -------------------------
    # Save
    # -------------------------
    df.to_csv(OUT, index=False)

    print(f"\n✅ Saved calibrated board → {OUT}")
    print("\n🔥 Top 10 calibrated edges (highest → lowest):")
    print(df[["player", "prop", "book_line", "true_projection", "true_edge"]].head(10))


if __name__ == "__main__":
    main()