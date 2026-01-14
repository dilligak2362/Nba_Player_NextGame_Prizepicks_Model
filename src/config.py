from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

RAW_BOXSCORES_CSV = RAW_DIR / "historical_boxscores.csv"
PROCESSED_DATASET_CSV = PROCESSED_DIR / "player_game_dataset.csv"

for p in [RAW_DIR, PROCESSED_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)
