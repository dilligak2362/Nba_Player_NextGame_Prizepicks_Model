import pandas as pd
from pathlib import Path

BOARD_PATH = Path("data/processed/calibrated_board.csv")
OUT_PATH = Path("data/calibration_parlays/calibration_parlay_slips.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_parlays(df, max_legs=6):
    df = df.sort_values("true_edge_for_pick", ascending=False).reset_index(drop=True)
    n = len(df)

    print(f"\n📊 Total calibration props: {n}")
    print("🔗 Building calibration parlays (top edge → down)\n")

    parlays = []

    for legs in range(2, max_legs + 1):
        max_slips = n // legs
        print(f"➡️ Building {legs}-leg parlays (target: {max_slips})")

        i = 0
        count = 0

        while i + legs <= n:
            slip = df.iloc[i:i+legs]

            total_edge = slip["true_edge_for_pick"].sum()

            parlays.append({
                "legs": legs,
                "players": " | ".join(slip["player"].values),
                "props": " | ".join(slip["prop"].values),
                "directions": " | ".join(slip["direction"].values),
                "edges": " | ".join(slip["true_edge_for_pick"].round(3).astype(str).values),
                "total_edge": round(total_edge, 4),
            })

            count += 1
            i += legs

        print(f"   ✅ Finished {legs}-leg parlays → {count} slips\n")

    return pd.DataFrame(parlays)


def main():
    print("\n📥 Loading calibrated board...")
    df = pd.read_csv(BOARD_PATH)

    required = {"player", "prop", "direction", "true_edge_for_pick"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["true_edge_for_pick"] = pd.to_numeric(df["true_edge_for_pick"], errors="coerce")
    df = df.dropna(subset=["true_edge_for_pick"])

    parlays = build_parlays(df, max_legs=6)
    parlays.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Saved calibration parlay slips → {OUT_PATH}")
    print(f"Total slips: {len(parlays)}")
    print("\n🔥 Sample slips:")
    print(parlays.head(10))


if __name__ == "__main__":
    main()