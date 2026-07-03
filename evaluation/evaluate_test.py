import joblib
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "models/best_model.pkl"

X_TEST = "training/training_data/X_test.csv"
Y_TEST = "training/training_data/y_test.csv"

ENCODER_PATH = "models/label_encoder.pkl"

# ==========================================================

print("=" * 70)
print("TEST DATA PREDICTION")
print("=" * 70)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# ==========================================================
# LOAD TEST DATA
# ==========================================================

X_test = pd.read_csv(X_TEST)
y_test = pd.read_csv(Y_TEST).iloc[:, 0]

print("\nTesting Samples :", X_test.shape)

# ==========================================================
# PREDICTION
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# CONVERT LABELS TO ORIGINAL ATTACK NAMES
# ==========================================================

actual = encoder.inverse_transform(y_test)
predicted = encoder.inverse_transform(y_pred)

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