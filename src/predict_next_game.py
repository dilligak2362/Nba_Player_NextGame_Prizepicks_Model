import pandas as pd
from joblib import load
from pathlib import Path

DATA_PATH = Path("data/processed/player_game_dataset.csv")

MODEL_DIR = Path("models")
PTS_MODEL = MODEL_DIR / "pts_rf.joblib"
REB_MODEL = MODEL_DIR / "reb_rf.joblib"
AST_MODEL = MODEL_DIR / "ast_rf.joblib"
STL_MODEL = MODEL_DIR / "stl_rf.joblib"
BLK_MODEL = MODEL_DIR / "blk_rf.joblib"


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Detect player column
PLAYER_COL = None
for col in ["player_name", "player", "name"]:
    if col in df.columns:
        PLAYER_COL = col
        break

if PLAYER_COL is None:
    raise Exception(f"No player column found. Columns are:\n{list(df.columns)}")

print(f"Using player column: {PLAYER_COL}")


print("Loading models...")
pts_model = load(PTS_MODEL)
reb_model = load(REB_MODEL)
ast_model = load(AST_MODEL)
stl_model = load(STL_MODEL)
blk_model = load(BLK_MODEL)


############################################
# Get Player Row
############################################
def get_player_input():
    name = input("\nEnter PLAYER name (e.g., LeBron James): ").strip().lower()

    matches = df[df[PLAYER_COL].str.lower().str.contains(name)]

    if matches.empty:
        print("\n❌ Player not found in dataset")
        return None

    player_games = matches.sort_values("date")

    latest = player_games.iloc[-1]

    print(f"\nFound Player: {latest[PLAYER_COL]} ({latest['team']})")
    return player_games


############################################
# Build Features from Last 5 Games
############################################
def build_last5_features(player_games):

    last5 = player_games.tail(5)

    feats = {
        "pts_avg_last5": last5["pts"].mean(),
        "reb_avg_last5": last5["reb"].mean(),
        "ast_avg_last5": last5["ast"].mean(),
        "blk_avg_last5": last5["blk"].mean(),
        "stl_avg_last5": last5["stl"].mean(),
        "minutes_avg_last5": last5["min"].replace(":", ".", regex=True).astype(float).mean()
        if "min" in last5.columns
        else 30.0
    }

    print("\nUsing these last-5 averages as features:")
    for k,v in feats.items():
        print(f"{k}: {round(v,2)}")

    return feats


############################################
# Predict
############################################
def predict_next_game(player_games):

    feats = build_last5_features(player_games)

    X = [[
        feats["pts_avg_last5"],
        feats["reb_avg_last5"],
        feats["ast_avg_last5"],
        feats["blk_avg_last5"],
        feats["stl_avg_last5"],
        feats["minutes_avg_last5"]
    ]]

    preds = {
        "PTS": round(float(pts_model.predict(X)[0]), 2),
        "REB": round(float(reb_model.predict(X)[0]), 2),
        "AST": round(float(ast_model.predict(X)[0]), 2),
        "STL": round(float(stl_model.predict(X)[0]), 2),
        "BLK": round(float(blk_model.predict(X)[0]), 2),
    }

    return preds


############################################
# Main
############################################
def main():
    player_games = get_player_input()
    if player_games is None:
        return

    preds = predict_next_game(player_games)

    last = player_games.iloc[-1]

    print("\n==============================")
    print(" NEXT GAME PREDICTIONS ")
    print("==============================")
    print(f"Player: {last[PLAYER_COL]}")
    print(f"Team: {last['team']} vs {last['opponent']}")
    print("------------------------------")
    for stat, value in preds.items():
        print(f"{stat}: {value}")
    print("==============================\n")


if __name__ == "__main__":
    main()


