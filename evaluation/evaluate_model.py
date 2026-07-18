import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "models/gradient_boosting.pkl"

X_TEST_PATH = "training/training_data/X_test.csv"
Y_TEST_PATH = "training/training_data/y_test.csv"

# ==========================================================

print("=" * 70)
print("CATBOOST MODEL EVALUATION")
print("=" * 70)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(MODEL_PATH)

# ==========================================================
# LOAD TEST DATA
# ==========================================================

X_test = pd.read_csv(X_TEST_PATH)

y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

# Load label encoder to get class names
le = joblib.load("models/label_encoder.pkl")
y_test_encoded = y_test
class_names = [str(c) for c in le.classes_]

print("\nTesting Samples :", X_test.shape)

# ==========================================================
# PREDICTION
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(y_test_encoded, y_pred)

precision_macro = precision_score(
    y_test_encoded,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test_encoded,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test_encoded,
    y_pred,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_test_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)

# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy           : {accuracy:.4f}")

print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1-Score     : {f1_macro:.4f}")

print()

print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1-Score  : {f1_weighted:.4f}")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test_encoded, y_pred)

print(cm)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_test_encoded,
        y_pred,

        labels=range(len(class_names)),

        target_names=class_names,

        digits=4,

        zero_division=0

    )

)

print("\n" + "=" * 70)
print("CATBOOST MODEL EVALUATION COMPLETED")
print("=" * 70)