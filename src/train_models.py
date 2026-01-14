import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

DATA = Path("data/processed/training_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ADD TO HERE
TARGETS = ["pts", "reb", "ast", "stl", "blk", "to"]

print("Loading training dataset...")
df = pd.read_csv(DATA)
print("Training rows (raw):", len(df))

# Add TO features
base_features = [
    "roll_avg_pts", "roll_avg_reb", "roll_avg_ast",
    "roll_avg_stl", "roll_avg_blk", "roll_avg_to",
    "lag1_pts", "lag1_reb", "lag1_ast", "lag1_stl", "lag1_blk", "lag1_to",
    "usage_proxy", "lag1_usage", "roll_usage", "ewm_usage",
    "opp_allow_pts", "opp_allow_reb", "opp_allow_ast",
    "opp_allow_stl", "opp_allow_blk", "opp_allow_to",
    "opp_pace_proxy",
    "games_played", "min", "is_home",
]

train_df = df.copy()

# Validate targets exist
for t in TARGETS:
    target_col = f"target_{t}"
    if target_col not in train_df.columns:
        raise Exception(f"Missing target column {target_col} in dataset")

print("Dropping rows without any targets...")
train_df = train_df.dropna(subset=[f"target_{t}" for t in TARGETS])

print("Rows after dropping missing targets:", len(train_df))
print()

for t in TARGETS:
    print(f"============ TRAINING {t.upper()} MODEL ============")

    target_col = f"target_{t}"

    feature_cols = [c for c in base_features if c in train_df.columns]

    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    print(f"Using {X.shape[0]} rows, {X.shape[1]} features")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )

    model.fit(X, y)

    preds = model.predict(X)
    mae = (abs(preds - y)).mean()
    print(f"MAE: {mae:.2f}")

    out_path = MODEL_DIR / f"{t}_rf.joblib"
    dump(model, out_path)
    print(f"Saved → {out_path}\n")

print("✅ ALL MODELS TRAINED SUCCESSFULLY")



