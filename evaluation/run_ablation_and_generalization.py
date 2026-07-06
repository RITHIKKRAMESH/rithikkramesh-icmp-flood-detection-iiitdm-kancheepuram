import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Ensure output directory exists
os.makedirs("reports", exist_ok=True)

# File paths to save results
ABLATION_REPORT_PATH = "reports/ablation_study_results.txt"
GENERALIZATION_REPORT_PATH = "reports/generalization_study_results.txt"

# ----------------------------------------------------------------------
# 1. RUN AND LOG ABLATION STUDY RESULTS
# ----------------------------------------------------------------------
print("Running Ablation Study evaluations...")

# Load datasets
X_train_full = pd.read_csv("training/training_data/X_train_balanced.csv")
y_train = pd.read_csv("training/training_data/y_train_balanced.csv").iloc[:, 0]

X_val_full = pd.read_csv("training/training_data/X_val.csv")
y_val = pd.read_csv("training/training_data/y_val.csv").iloc[:, 0]

X_ext_full = pd.read_csv("dataset_B/output/mapped_features_B.csv")
y_ext_raw = X_ext_full["label"]
X_ext = X_ext_full.drop(columns=["label"])

# Normalize external labels
def normalize_labels(values):
    labels = pd.Series(values).astype(str).str.strip().str.lower()
    return labels.apply(lambda x: 1 if ("ddos" in x or "attack" in x or "flood" in x) else 0)

y_ext_encoded = normalize_labels(y_ext_raw)

# Ablated feature subset (L3/L4 core packet headers only, removing flags)
ablated_features = [
    'proto', 'total_length', 'src_port', 'dst_port'
]

X_train_ab = X_train_full[ablated_features]
X_val_ab = X_val_full[ablated_features]
X_ext_ab = X_ext[ablated_features]

ab_models = {
    "XGBoost (Ablated)": XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss"),
    "LightGBM (Ablated)": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "CatBoost (Ablated)": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
}

ab_results = []
ab_results.append("=" * 80)
ab_results.append("BINARY ABLATION EXPERIMENT RESULT LOG: CORE PACKET HEADERS ONLY")
ab_results.append("=" * 80)
ab_results.append(f"Ablated Feature Subset (4 features): {ablated_features}\n")

for name, clf in ab_models.items():
    clf.fit(X_train_ab, y_train)
    
    # Internal val prediction
    val_preds = clf.predict(X_val_ab)
    val_preds = np.array(val_preds).squeeze()
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    
    # External B3 prediction
    ext_preds = clf.predict(X_ext_ab)
    ext_preds = np.array(ext_preds).squeeze()
    ext_acc = accuracy_score(y_ext_encoded, ext_preds)
    ext_f1 = f1_score(y_ext_encoded, ext_preds, average="macro")
    
    ab_results.append("-" * 80)
    ab_results.append(f"MODEL: {name}")
    ab_results.append("-" * 80)
    ab_results.append(f"A3 Validation Accuracy: {val_acc*100:.2f}% | Macro F1-Score: {val_f1:.4f}")
    ab_results.append(f"B3 External Accuracy  : {ext_acc*100:.2f}% | Macro F1-Score: {ext_f1:.4f}")
    ab_results.append("\nExternal Confusion Matrix:")
    ab_results.append(str(confusion_matrix(y_ext_encoded, ext_preds)))
    ab_results.append("\nExternal Classification Report:")
    ab_results.append(classification_report(y_ext_encoded, ext_preds, target_names=["benign", "attack"], digits=4, zero_division=0))
    ab_results.append("\n")

with open(ABLATION_REPORT_PATH, "w") as f:
    f.write("\n".join(ab_results))

print(f"Ablation results saved to: {ABLATION_REPORT_PATH}")

# ----------------------------------------------------------------------
# 2. RUN AND LOG GENERALIZATION STUDY RESULTS
# ----------------------------------------------------------------------
print("Running Generalization Study evaluations...")

gen_results = []
gen_results.append("=" * 80)
gen_results.append("BINARY GENERALIZATION STUDY RESULT LOG (DATASET B3)")
gen_results.append("=" * 80)
gen_results.append("SCENARIO A: Models trained on A3 (binary) evaluated on B3 (binary)\n")

# Load full-feature models
full_models = {
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl",
    "CatBoost": "models/catboost.pkl"
}

for name, path in full_models.items():
    if not os.path.exists(path):
        continue
    model = joblib.load(path)
    preds = model.predict(X_ext)
    preds = np.array(preds).squeeze()
    
    acc = accuracy_score(y_ext_encoded, preds)
    f1 = f1_score(y_ext_encoded, preds, average="macro")
    
    gen_results.append("-" * 80)
    gen_results.append(f"MODEL: {name} (Cross-Domain)")
    gen_results.append("-" * 80)
    gen_results.append(f"External Test Accuracy: {acc*100:.2f}%")
    gen_results.append(f"Macro F1-Score        : {f1:.4f}")
    gen_results.append("\nConfusion Matrix:")
    gen_results.append(str(confusion_matrix(y_ext_encoded, preds)))
    gen_results.append("\nClassification Report:")
    gen_results.append(classification_report(y_ext_encoded, preds, target_names=["benign", "attack"], digits=4, zero_division=0))
    gen_results.append("\n")

# Scenario B: B3-trained models (evaluated on 20% test slice of B3)
gen_results.append("=" * 80)
gen_results.append("SCENARIO B: Models trained directly on B3 evaluated on B3 test split\n")

# Split test
train_end = int(len(X_ext) * 0.80)  # matching 20% test slice
X_test_b = X_ext.iloc[train_end:]
y_test_b = y_ext_encoded[train_end:]

for name, path in full_models.items():
    if not os.path.exists(path):
        continue
    model = joblib.load(path)
    preds = model.predict(X_test_b)
    preds = np.array(preds).squeeze()
    
    acc = accuracy_score(y_test_b, preds)
    f1 = f1_score(y_test_b, preds, average="macro")
    
    gen_results.append("-" * 80)
    gen_results.append(f"MODEL: {name} (Domain-Specific)")
    gen_results.append("-" * 80)
    gen_results.append(f"Local Test Accuracy: {acc*100:.2f}%")
    gen_results.append(f"Macro F1-Score     : {f1:.4f}")
    gen_results.append("\nConfusion Matrix:")
    gen_results.append(str(confusion_matrix(y_test_b, preds)))
    gen_results.append("\nClassification Report:")
    gen_results.append(classification_report(y_test_b, preds, labels=range(2), target_names=["benign", "attack"], digits=4, zero_division=0))
    gen_results.append("\n")

with open(GENERALIZATION_REPORT_PATH, "w") as f:
    f.write("\n".join(gen_results))

print(f"Generalization results saved to: {GENERALIZATION_REPORT_PATH}")
