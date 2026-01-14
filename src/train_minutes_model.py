import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

DATA = Path("data/processed/minutes_training_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

print("Loading minutes training dataset...")
df = pd.read_csv(DATA)
print("Rows:", len(df))

features = [
    "roll_min", "lag1_min", "lag2_min", "ewm_min",
    "usage_proxy", "roll_usage",
    "is_star", "is_starter", "is_rotation",
    "games_played"
]

df = df.dropna(subset=["target_min"])

X = df[features]
y = df["target_min"]

mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

print("Training rows:", len(X))

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=3,
    n_jobs=-1
)

model.fit(X, y)

preds = model.predict(X)
mae = abs(preds - y).mean()

print(f"Minutes MAE: {mae:.2f}")

out = MODEL_DIR / "minutes_rf.joblib"
dump(model, out)

print(f"Saved minutes model → {out}")
