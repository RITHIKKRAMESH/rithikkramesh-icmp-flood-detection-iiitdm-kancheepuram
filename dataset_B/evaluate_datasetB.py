import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==========================================================
# PATHS
# ==========================================================

MAPPED_DATASET_PATH = "dataset_B/output/mapped_features_B.csv"
ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_DIR = "dataset_B/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("EXTERNAL DATASET B EVALUATION FOR ALL MODELS")
print("=" * 70)

# ==========================================================
# LOAD DATA AND ENCODER
# ==========================================================

encoder = joblib.load(ENCODER_PATH)
df = pd.read_csv(MAPPED_DATASET_PATH)

if "label" not in df.columns:
    raise ValueError("Mapped dataset must contain the 'label' column.")

# Separate features and labels
X = df.drop(columns=["label"])
y_true_raw = df["label"]

def normalize_labels(values):
    labels = pd.Series(values).astype(str).str.strip().str.lower()
    return labels.apply(lambda x: 1 if ("ddos" in x or "attack" in x or "flood" in x) else 0)

y_true_encoded = normalize_labels(y_true_raw)
y_true_binary = y_true_encoded

print(f"\nDataset B Shape        : {df.shape}")
print(f"Features Shape         : {X.shape}")
print(f"Unique classes in true: {y_true_encoded.unique()}")
print("Encoder classes        : ['benign', 'attack']")

# Models to evaluate
models_dict = {
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl",
    "CatBoost": "models/catboost.pkl"
}

for model_name, model_path in models_dict.items():
    if not os.path.exists(model_path):
        print(f"\nModel {model_name} not found at {model_path}, skipping.")
        continue
        
    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 70)
    
    model = joblib.load(model_path)
    y_pred_encoded = model.predict(X)
    import numpy as np
    y_pred_encoded = np.array(y_pred_encoded).squeeze()
    
    y_pred_binary = y_pred_encoded
    
    # -----------------------------
    # Binary Metrics
    # -----------------------------
    binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    binary_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    binary_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    print("\n[BINARY PERFORMANCE (Benign vs Attack)]")
    print(f"Accuracy        : {binary_accuracy:.4f}")
    print(f"Precision       : {binary_precision:.4f}")
    print(f"Recall          : {binary_recall:.4f}")
    print(f"F1-Score        : {binary_f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_binary, y_pred_binary))
    
    print("\nClassification Report:")
    print(classification_report(y_true_binary, y_pred_binary, target_names=["Benign", "Attack"], digits=4, zero_division=0))
    
    # Save predictions for this model
    class_map = {0: "benign", 1: "attack"}
    pred_df = pd.DataFrame({
        "Actual_Label": pd.Series(y_true_binary).map(class_map),
        "Predicted_Label": pd.Series(y_pred_binary).map(class_map),
        "Actual_Binary": y_true_binary,
        "Predicted_Binary": y_pred_binary
    })
    
    save_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_predictions.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"\nPredictions saved to: {save_path}")

print("\n" + "=" * 70)
print("ALL EVALUATIONS COMPLETED")
print("=" * 70)
