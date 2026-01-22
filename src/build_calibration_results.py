import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# -----------------------
# PATHS
# -----------------------
SETTLED_DIR = Path("data/history/settled")
CALIBRATED_BOARD_PATH = Path("data/processed/calibrated_board.csv")

OUT_DIR = Path("data/calibration_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# CLEANING HELPERS
# -----------------------
def clean_player(s: str) -> str:
    if pd.isna(s):
        return ""
    return " ".join(str(s).strip().split())


def clean_prop(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def build_key(df: pd.DataFrame) -> pd.Series:
    return df["player"].apply(clean_player) + "|" + df["prop"].apply(clean_prop)


# -----------------------
# MAIN
# -----------------------
def main(target_date: str = None):
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    settled_path = SETTLED_DIR / f"settled_{target_date}.csv"

    if not CALIBRATED_BOARD_PATH.exists():
        raise FileNotFoundError(f"Missing calibrated board: {CALIBRATED_BOARD_PATH}")

    if not settled_path.exists():
        raise FileNotFoundError(f"Missing settled file: {settled_path}")

    print("\n📥 Loading settled results...")
    settled = pd.read_csv(settled_path)

    print("📥 Loading calibrated board...")
    calib = pd.read_csv(CALIBRATED_BOARD_PATH)

    # -----------------------
    # NORMALIZATION
    # -----------------------
    settled["player"] = settled["player"].apply(clean_player)
    settled["prop"] = settled["prop"].apply(clean_prop)
    settled["direction"] = settled["direction"].astype(str).str.strip().str.upper()

    calib["player"] = calib["player"].apply(clean_player)
    calib["prop"] = calib["prop"].apply(clean_prop)

    # -----------------------
    # BUILD KEYS
    # -----------------------
    settled["key"] = build_key(settled)
    calib["key"] = build_key(calib)

    # -----------------------
    # DEDUPE CALIB BOARD
    # -----------------------
    calib = (
        calib.sort_values("true_edge", ascending=False)
             .drop_duplicates("key", keep="first")
    )

    # -----------------------
    # MERGE
    # -----------------------
    merged = settled.merge(
        calib[["key", "true_projection", "true_edge"]],
        on="key",
        how="left"
    )

    # -----------------------
    # DIRECTION-AWARE EDGE
    # -----------------------
    merged["true_edge_for_pick"] = np.where(
        merged["direction"] == "UNDER",
        merged["book_line"] - merged["true_projection"],
        merged["true_projection"] - merged["book_line"]
    )

    # -----------------------
    # OUTPUT TABLE
    # -----------------------
    out = merged[[
        "player",
        "prop",
        "book_line",
        "direction",
        "model_prediction",
        "true_projection",
        "edge",
        "true_edge",
        "true_edge_for_pick",
        "actual_value",
        "result"
    ]].copy()

    # ✅ SORT BY TRUE EDGE (BEST FIRST)
    out = out.sort_values("true_edge_for_pick", ascending=False).reset_index(drop=True)

    # -----------------------
    # SAVE
    # -----------------------
    out_path = OUT_DIR / f"calibration_results_{target_date}.csv"
    out.to_csv(out_path, index=False)

    print(f"\n✅ Calibration audit saved → {out_path}")

    # -----------------------
    # PERFORMANCE SUMMARY
    # -----------------------
    print("\nResult counts:")
    print(out["result"].value_counts(dropna=False).to_string())

    graded = out[out["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    graded["is_win"] = (graded["result"] == "WIN").astype(int)
    graded["is_loss"] = (graded["result"] == "LOSS").astype(int)

    if len(graded) > 0:
        win_rate = graded["is_win"].sum() / max(1, (graded["is_win"].sum() + graded["is_loss"].sum()))
        avg_edge = graded["true_edge_for_pick"].mean()

        print("\n📊 Calibration Performance (direction-aware):")
        print(f" - Win rate (ex push): {win_rate:.4f}")
        print(f" - Avg true_edge_for_pick: {avg_edge:.4f}")

        print("\n🔥 Top 10 calibrated edges:")
        print(out[[
            "player", "prop", "direction",
            "book_line", "true_projection",
            "true_edge_for_pick", "actual_value", "result"
        ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()