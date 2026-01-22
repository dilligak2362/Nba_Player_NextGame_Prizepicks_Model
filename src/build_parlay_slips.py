import pandas as pd
from pathlib import Path

BOARD_PATH = Path("data/processed/merged_props_predictions.csv")
OUT_PATH = Path("data/processed/parlay_slips.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_parlays(df, max_legs=6):
    df = df.sort_values("edge").reset_index(drop=True)
    n = len(df)

    print(f"\n📊 Total board props: {n}")
    print("🔗 Building block-grouped parlays (bottom → top)\n")

    parlays = []

    for legs in range(2, max_legs + 1):
        max_slips = n // legs
        print(f"➡️ Building {legs}-leg parlays (target: {max_slips})")

        i = 0
        count = 0

        while i + legs <= n:
            slip = df.iloc[i:i+legs]

            total_edge = slip["edge"].sum()

            parlays.append({
                "legs": legs,
                "players": " | ".join(slip["player"].values),
                "props": " | ".join(slip["prop"].values),
                "edges": " | ".join(slip["edge"].round(2).astype(str).values),
                "total_edge": round(total_edge, 3),
            })

            count += 1
            i += legs

            if count % 10 == 0 or count == max_slips:
                print(f"   ⏳ {legs}-leg progress: {count}/{max_slips}")

        print(f"   ✅ Finished {legs}-leg parlays → {count} slips\n")

    return pd.DataFrame(parlays)


def main():
    print("\n📥 Loading daily board...")
    df = pd.read_csv(BOARD_PATH)

    required = {"player", "prop", "edge"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["edge"] = pd.to_numeric(df["edge"], errors="coerce")
    df = df.dropna(subset=["edge"])

    parlays = build_parlays(df, max_legs=6)

    parlays.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Saved parlay slips → {OUT_PATH}")
    print(f"Total slips: {len(parlays)}")
    print("\n🔥 Sample slips:")
    print(parlays.head(10))


if __name__ == "__main__":
    main()