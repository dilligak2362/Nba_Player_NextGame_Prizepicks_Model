import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump

DATA = Path("data/processed/calibration_dataset.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

print("Loading calibration dataset...")
df = pd.read_csv(DATA)

# -------------------------
# Target
# -------------------------
target = "actual"

# -------------------------
# Only keep numeric features
# -------------------------
drop_cols = ["actual", "player", "prop", "game_date"]
features = [c for c in df.columns if c not in drop_cols]

X = df[features]
y = df[target]

# Drop any remaining NaNs
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

print("Training rows:", len(X))
print("Features:", list(X.columns))

# -------------------------
# Train model
# -------------------------
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X, y)

# -------------------------
# Evaluation
# -------------------------
preds = model.predict(X)
mae = abs(preds - y).mean()

print(f"Calibration MAE: {mae:.3f}")

# -------------------------
# Save model
# -------------------------
out = MODEL_DIR / "calibration_model.joblib"
dump(model, out)

print(f"✅ Saved calibration model → {out}")
print(f"✅ Saved calibration model → {out}")