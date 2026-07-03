from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"


if not MODEL_PATH.exists():
	raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully!")
print(model)