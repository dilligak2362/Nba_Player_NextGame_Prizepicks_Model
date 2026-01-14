import re
import pandas as pd
from pathlib import Path

BOARDS_DIR = Path("data/history/boards")
SETTLED_DIR = Path("data/history/settled")
OUT = Path("data/processed/calibration_dataset.csv")

OUT.parent.mkdir(parents=True, exist_ok=True)


def _extract_date_from_name(path: Path) -> str | None:
    """
    board_YYYY-MM-DD.csv or settled_YYYY-MM-DD.csv -> YYYY-MM-DD
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return m.group(1) if m else None


def _load_with_date(files: list[Path], kind: str) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            d = _extract_date_from_name(f)
            df["game_date"] = d
            df["__src_file"] = f.name
            df["__kind"] = kind
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Failed reading {f}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # normalize expected fields if present
    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip()

    # some boards may have 'stat' instead of 'prop'
    if "prop" not in df.columns and "stat" in df.columns:
        df = df.rename(columns={"stat": "prop"})

    if "prop" in df.columns:
        df["prop"] = df["prop"].astype(str).str.upper().str.strip()

    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.upper().str.strip()

    if "book_line" in df.columns:
        df["book_line"] = pd.to_numeric(df["book_line"], errors="coerce")

    if "model_prediction" in df.columns:
        df["model_prediction"] = pd.to_numeric(df["model_prediction"], errors="coerce")

    if "edge" in df.columns:
        df["edge"] = pd.to_numeric(df["edge"], errors="coerce")

    if "proj_min" in df.columns:
        df["proj_min"] = pd.to_numeric(df["proj_min"], errors="coerce")

    if "actual_value" in df.columns:
        df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")

    return df


def main():
    board_files = sorted(BOARDS_DIR.glob("board_*.csv"))
    settled_files = sorted(SETTLED_DIR.glob("settled_*.csv"))

    if not board_files or not settled_files:
        print("❌ Missing boards or settled history. Need at least 1 of each.")
        return

    boards = _normalize(_load_with_date(board_files, kind="board"))
    settled = _normalize(_load_with_date(settled_files, kind="settled"))

    if boards.empty or settled.empty:
        print("❌ Boards or settled loaded empty.")
        return

    # Validate minimum required columns
    needed_boards = {"game_date", "player", "prop", "book_line", "direction"}
    needed_settled = {"game_date", "player", "prop", "book_line", "direction", "actual_value"}

    miss_b = needed_boards - set(boards.columns)
    miss_s = needed_settled - set(settled.columns)

    if miss_b:
        raise ValueError(f"Boards missing required columns: {miss_b}")
    if miss_s:
        raise ValueError(f"Settled missing required columns: {miss_s}")

    # Merge by date to prevent cross-day collisions
    df = boards.merge(
        settled,
        on=["game_date", "player", "prop", "book_line", "direction"],
        how="inner",
        suffixes=("_board", "_settled"),
    )

    if df.empty:
        print("⚠️ No matches between boards and settled. Check naming/date alignment.")
        return

    # --- choose board-side features (what you knew before the game) ---
    # model_prediction + edge should come from the board snapshot
    mp_col = "model_prediction_board" if "model_prediction_board" in df.columns else "model_prediction"
    edge_col = "edge_board" if "edge_board" in df.columns else "edge"

    # proj_min is optional (depends on your pipeline)
    if "proj_min_board" in df.columns:
        proj_min = df["proj_min_board"]
    elif "proj_min" in df.columns:
        proj_min = df["proj_min"]
    else:
        proj_min = 0

    cal = pd.DataFrame({
        "game_date": df["game_date"],
        "player": df["player"],
        "prop": df["prop"],
        "model_prediction": pd.to_numeric(df.get(mp_col, pd.NA), errors="coerce"),
        "book_line": pd.to_numeric(df["book_line"], errors="coerce"),
        "proj_min": pd.to_numeric(proj_min, errors="coerce"),
        "edge": pd.to_numeric(df.get(edge_col, pd.NA), errors="coerce"),
        "stat": df["prop"],  # stat == prop type (PTS/REB/PRA/etc)
        "actual": pd.to_numeric(df["actual_value"], errors="coerce"),
    })

    # Drop junk rows
    cal = cal.dropna(subset=["model_prediction", "book_line", "actual"]).copy()

    # One-hot stat
    cal = pd.get_dummies(cal, columns=["stat"], prefix="stat")

    cal.to_csv(OUT, index=False)
    print(f"✅ Saved calibration dataset → {OUT}")
    print(f"Rows: {len(cal)}")


if __name__ == "__main__":
    main()