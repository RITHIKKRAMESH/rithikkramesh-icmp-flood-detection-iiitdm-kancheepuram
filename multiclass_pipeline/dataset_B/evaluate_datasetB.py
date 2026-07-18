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
ENCODER_PATH = "multiclass_pipeline/models/label_encoder.pkl"
OUTPUT_DIR = "multiclass_pipeline/dataset_B/output"

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



# Encode labels using the encoder from training
y_true_encoded = encoder.transform(y_true_labels)

print(f"\nDataset B Shape        : {df.shape}")
print(f"Features Shape         : {X.shape}")
print(f"Unique classes in true: {y_true_labels.unique()}")
print(f"Encoder classes        : {encoder.classes_}")

# Models to evaluate
models_dict = {
    "XGBoost": "multiclass_pipeline/models/xgboost.pkl",
    "LightGBM": "multiclass_pipeline/models/lightgbm.pkl",
    "CatBoost": "multiclass_pipeline/models/catboost.pkl",
    "DecisionTree": "multiclass_pipeline/models/decision_tree.pkl",
    "AdaBoost": "multiclass_pipeline/models/adaboost.pkl",
    "LogisticRegression": "multiclass_pipeline/models/logistic_regression.pkl",
    "GradientBoosting": "multiclass_pipeline/models/gradient_boosting.pkl",
    "MLP": "multiclass_pipeline/models/mlp.pkl"
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
    
    # Save predictions for this model
    pred_df = pd.DataFrame({
        "Actual_Label": y_true_labels,
        "Predicted_Label": y_pred_labels,
        "Actual_Encoded": y_true_encoded,
        "Predicted_Encoded": y_pred_encoded
    })
    
    save_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_predictions.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"\nPredictions saved to: {save_path}")

print("\n" + "=" * 70)
print("ALL EVALUATIONS COMPLETED")
print("=" * 70)
