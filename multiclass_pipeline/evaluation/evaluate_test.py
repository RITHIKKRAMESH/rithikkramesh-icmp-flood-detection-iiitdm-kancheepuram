import os

import joblib
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "models/best_model.pkl"

X_TEST = "training/training_data/X_test.csv"
Y_TEST = "training/training_data/y_test.csv"

ENCODER_PATH = "models/label_encoder.pkl"
MAPPED_DATASET_B = "dataset_B/output/mapped_features_B.csv"

# ==========================================================

print("=" * 70)
print("TEST DATA PREDICTION")
print("=" * 70)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

expected_features = None
if hasattr(model, "get_booster"):
    booster_features = model.get_booster().feature_names
    if booster_features:
        expected_features = list(booster_features)
elif hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)

# ==========================================================
# LOAD TEST DATA
# ==========================================================

X_test = pd.read_csv(X_TEST)
y_test = pd.read_csv(Y_TEST).iloc[:, 0]

if expected_features is not None and list(X_test.columns) != expected_features:
    if os.path.exists(MAPPED_DATASET_B):
        candidate = pd.read_csv(MAPPED_DATASET_B)
        if "label" in candidate.columns:
            print("\nFeature mismatch detected. Switching to mapped Dataset B test data.")
            X_test = candidate.drop(columns=["label"])
            y_test = candidate["label"]
        else:
            raise ValueError(
                "Test features do not match the model schema, and the fallback dataset "
                f"{MAPPED_DATASET_B} does not contain a label column."
            )
    else:
        raise ValueError(
            "Test features do not match the model schema. Regenerate training/training_data/X_test.csv "
            "from feature_engineering/output/Features_A.csv or provide the mapped Dataset B file."
        )

if expected_features is not None:
    X_test = X_test.reindex(columns=expected_features, fill_value=0)

print("\nTesting Samples :", X_test.shape)

# ==========================================================
# PREDICTION
# ==========================================================

y_pred = model.predict(X_test)
y_pred = np.asarray(y_pred).squeeze()

# ==========================================================
# CONVERT LABELS TO ORIGINAL ATTACK NAMES
# ==========================================================

if pd.api.types.is_numeric_dtype(y_test):
    actual = encoder.inverse_transform(y_test.astype(int).to_numpy())
else:
    actual = y_test.astype(str).to_numpy()

predicted = encoder.inverse_transform(np.asarray(y_pred, dtype=int))

# ==========================================================
# CREATE RESULT DATAFRAME
# ==========================================================

results = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted
})

results["Correct"] = results["Actual"] == results["Predicted"]

# ==========================================================
# SUMMARY
# ==========================================================

correct = results["Correct"].sum()
wrong = len(results) - correct

accuracy = (correct / len(results)) * 100

print("\nCorrect Predictions :", correct)
print("Wrong Predictions   :", wrong)
print(f"Testing Accuracy    : {accuracy:.2f}%")

# ==========================================================
# SHOW FIRST 30 PREDICTIONS
# ==========================================================

print("\nFirst 30 Predictions")
print("-" * 70)

print(results.head(30))

# ==========================================================
# SAVE ALL PREDICTIONS
# ==========================================================

results.to_csv(
    "reports/test_predictions.csv",
    index=False
)

# ==========================================================
# SAVE ONLY WRONG PREDICTIONS
# ==========================================================

wrong_predictions = results[results["Correct"] == False]

wrong_predictions.to_csv(
    "reports/wrong_test_predictions.csv",
    index=False
)

print("\nReports Saved Successfully")

print("reports/test_predictions.csv")
print("reports/wrong_test_predictions.csv")

print("\n" + "=" * 70)
print("TEST PREDICTION COMPLETED")
print("=" * 70)