from __future__ import annotations

import argparse
import joblib
import pandas as pd

from config import PROCESSED_DATASET_CSV, MODEL_DIR
from train_models import FEATURE_COLS, TARGETS

def _load_models():
    models={}
    for t in TARGETS:
        path=MODEL_DIR / f"{t}_rf.joblib"
        if not path.exists():
            raise FileNotFoundError("Run training first")
        models[t]=joblib.load(path)
    return models

def predict_player_next_game(player_name:str, as_of_date:str|None=None):
    df=pd.read_csv(PROCESSED_DATASET_CSV, parse_dates=["date"])

    mask=df["player"].str.lower()==player_name.lower()
    if as_of_date:
        mask &= df["date"] < pd.to_datetime(as_of_date)

    player_df=df.loc[mask].sort_values("date")
    if player_df.empty:
        raise ValueError("Player not found in dataset")

    latest=player_df.iloc[-1]
    X=latest[FEATURE_COLS].astype(float).values.reshape(1,-1)

    models=_load_models()

    preds={}
    for t,m in models.items():
        preds[t]=float(m.predict(X)[0])
    return preds

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--player",required=True)
    parser.add_argument("--as_of")
    args=parser.parse_args()

    preds=predict_player_next_game(args.player,args.as_of)
    print(f"Predicted next-game stats for {args.player}:")
    print(f"Points : {preds['pts']:.1f}")
    print(f"Rebounds: {preds['reb']:.1f}")
    print(f"Assists : {preds['ast']:.1f}")

if __name__=="__main__":
    main()
