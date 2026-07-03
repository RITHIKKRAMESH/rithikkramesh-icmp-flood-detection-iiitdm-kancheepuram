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

# Normalize labels: map any normal variant to lowercase 'benign' to match label encoder
def normalize_labels(values):
    labels = pd.Series(values).astype(str).str.strip()
    return labels.replace(
        {
            "NORMAL_ICMP": "benign",
            "NORMAL_TCP": "benign",
            "NORMAL_UDP": "benign",
            "normal_icmp": "benign",
            "normal_tcp": "benign",
            "normal_udp": "benign",
            "normal": "benign",
            "BENIGN": "benign",
            "benign": "benign",
            "Benign": "benign"
        }
    )

y_true_labels = normalize_labels(y_true_raw)

# Binary labels: 0 for benign, 1 for attack
NORMAL_LABELS = {"benign", "normal"}
def to_binary_labels(values):
    normalized = pd.Series(values).astype(str).str.strip().str.lower()
    return (~normalized.isin(NORMAL_LABELS)).astype(int)

y_true_binary = to_binary_labels(y_true_labels)

# Encode labels using the encoder from training
y_true_encoded = encoder.transform(y_true_labels)

print(f"\nDataset B Shape        : {df.shape}")
print(f"Features Shape         : {X.shape}")
print(f"Unique classes in true: {y_true_labels.unique()}")
print(f"Encoder classes        : {encoder.classes_}")

# Models to evaluate
models_dict = {
    "Random Forest": "models/random_forest.pkl",
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
    
    # Invert predictions using label encoder
    y_pred_labels = encoder.inverse_transform(y_pred_encoded)
    y_pred_binary = to_binary_labels(y_pred_labels)
    
    # -----------------------------
    # Multiclass Metrics
    # -----------------------------
    multiclass_accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    multiclass_precision = precision_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)
    multiclass_recall = recall_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)
    multiclass_f1 = f1_score(y_true_encoded, y_pred_encoded, average="macro", zero_division=0)
    
    print("\n[MULTICLASS PERFORMANCE]")
    print(f"Accuracy        : {multiclass_accuracy:.4f}")
    print(f"Macro Precision : {multiclass_precision:.4f}")
    print(f"Macro Recall    : {multiclass_recall:.4f}")
    print(f"Macro F1-Score  : {multiclass_f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_encoded, y_pred_encoded))
    
    print("\nClassification Report:")
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=encoder.classes_, digits=4, zero_division=0))
    
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
    pred_df = pd.DataFrame({
        "Actual_Label": y_true_labels,
        "Predicted_Label": y_pred_labels,
        "Actual_Encoded": y_true_encoded,
        "Predicted_Encoded": y_pred_encoded,
        "Actual_Binary": y_true_binary,
        "Predicted_Binary": y_pred_binary
    })
    
    save_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_predictions.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"\nPredictions saved to: {save_path}")

print("\n" + "=" * 70)
print("ALL EVALUATIONS COMPLETED")
print("=" * 70)
