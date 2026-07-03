import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "models/lightgbm.pkl"
FEATURE_PATH = "feature_engineering/output/Features_A.csv"

# ==========================================================

print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

# Load Model
model = joblib.load(MODEL_PATH)

# Load Feature Names
X = pd.read_csv(FEATURE_PATH)

# Feature Importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({

    "Feature": X.columns,
    "Importance": importance

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop Features\n")

print(feature_importance)

# Save

os.makedirs(

    "feature_selection/output",

    exist_ok=True

)

feature_importance.to_csv(

    "feature_selection/output/feature_importance.csv",

    index=False

)

# Plot

plt.figure(figsize=(12,8))

plt.barh(

    feature_importance["Feature"][:20][::-1],

    feature_importance["Importance"][:20][::-1]

)

plt.xlabel("Importance")

plt.title("Top 20 Feature Importance (LightGBM)")

plt.tight_layout()

plt.savefig(

    "feature_selection/output/feature_importance.png"

)

plt.show()

print("\nFeature Importance Saved Successfully")