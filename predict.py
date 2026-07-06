import os
import joblib
import pandas as pd
import numpy as np

# ==========================================================
# CONFIGURATION
# ==========================================================

# Choose which dataset to draw the sample from:
# Option 1: "A1" (Local flow-level dataset)
# Option 2: "B1" (External packet-level/SDN dataset)
DATASET_SOURCE = "A3" 
SAMPLE_INDEX = 47  # Try index 47, 107, 133 for benign in A1!

# ==========================================================
# PATHS AND LOAD DATA
# ==========================================================

if DATASET_SOURCE == "A3":
    ENCODER_PATH = "models/label_encoder.pkl"
    models_dict = {
        "XGBoost": "models/xgboost.pkl",
        "LightGBM": "models/lightgbm.pkl",
        "CatBoost": "models/catboost.pkl"
    }
else:
    ENCODER_PATH = "models/label_encoder_B3.pkl"
    models_dict = {
        "XGBoost": "models/xgboost_B3.pkl",
        "LightGBM": "models/lightgbm_B3.pkl",
        "CatBoost": "models/catboost_B3.pkl"
    }

encoder = joblib.load(ENCODER_PATH)

print("=" * 70)
print(f"SINGLE SAMPLE INFERENCE TEST (Dataset Source: {DATASET_SOURCE}, Index: {SAMPLE_INDEX})")
print("=" * 70)

if DATASET_SOURCE == "A3":
    # Load processed training features to see the schema
    features_schema = pd.read_csv("feature_engineering/output/Features_A.csv", nrows=0).columns
    
    # Load the clean dataset A3
    df = pd.read_csv("preprocessing/cleaned_dataset/clean_dataset_A.csv")
    sample = df.iloc[SAMPLE_INDEX]
    
    actual_label_name = sample["label_binary"]
    
    # Separate features
    sample_features = sample.drop("label_binary")
    X_sample = pd.DataFrame([sample_features])[features_schema]

else:  # Dataset B3
    # Load mapped features of B3
    df = pd.read_csv("dataset_B/output/mapped_features_B.csv")
    sample = df.iloc[SAMPLE_INDEX]
    
    # Normalise raw labels in B3 to match training
    def normalize_label(val):
        val = str(val).strip()
        mapping = {
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
        return mapping.get(val, val)
        
    actual_label_name = normalize_label(sample["label"])
    
    # Separate features (excluding the label column)
    sample_features = sample.drop("label")
    X_sample = pd.DataFrame([sample_features])

# Convert all features to numeric format
X_sample = X_sample.apply(pd.to_numeric, errors="coerce").fillna(0)

# ==========================================================
# PRINT SAMPLE FEATURES
# ==========================================================
print("\n[Sample Features (Non-zero values only)]:")
print("-" * 50)
for col in X_sample.columns:
    val = X_sample.iloc[0][col]
    if val != 0:
         print(f"{col:<25}: {val}")

print("-" * 50)
class_names = ["benign", "attack"]
actual_label_str = class_names[int(actual_label_name)]
print(f"Actual Label             : {actual_label_str.upper()}")
print("=" * 70)

# ==========================================================
# RUN PREDICTION FOR ALL FOUR MODELS
# ==========================================================

print("\n[Model Predictions]:")
print("-" * 50)

for name, path in models_dict.items():
    if not os.path.exists(path):
        print(f"{name:<20} : Model file not found.")
        continue
        
    model = joblib.load(path)
    
    # Run prediction
    pred_encoded = model.predict(X_sample)
    
    # Ensure prediction is a 1D scalar
    pred_encoded = np.array(pred_encoded).squeeze()
    
    pred_label_str = class_names[int(pred_encoded)]
    status = "CORRECT" if pred_label_str.lower() == actual_label_str.lower() else "WRONG"
    print(f"{name:<20} : Predicted = {pred_label_str.upper():<12} | Status = {status}")

print("-" * 50)

# ==========================================================
# INDEX SUGGESTIONS FOR CONVENIENCE
# ==========================================================
print("\n[Suggested Indices for Testing]:")
print("-" * 50)
if DATASET_SOURCE == "A1":
    print("Dataset: preprocessing/cleaned_dataset/clean_dataset_A.csv")
    print("  - BENIGN    : [47, 107, 133, 158, 198]")
    print("  - DDOS_TCP  : [0, 2, 5, 8, 9]")
    print("  - DDOS_UDP  : [1, 7, 12, 17, 19]")
    print("  - DDOS_ICMP : [3, 4, 6, 10, 11]")
else:
    print("Dataset: dataset_B/output/mapped_features_B.csv")
    print("  - BENIGN    : [210000, 210001, 210002, 210003, 210004]")
    print("  - DDOS_ICMP : [0, 1, 2, 3, 4]")
    print("  - DDOS_TCP  : [70000, 70001, 70002, 70003, 70004]")
    print("  - DDOS_UDP  : [140000, 140001, 140002, 140003, 140004]")
print("-" * 50)