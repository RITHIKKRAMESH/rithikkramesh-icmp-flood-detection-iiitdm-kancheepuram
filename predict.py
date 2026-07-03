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
DATASET_SOURCE = "B1" 
SAMPLE_INDEX = 25  # The row index of the sample you want to test

# ==========================================================
# PATHS AND LOAD DATA
# ==========================================================

if DATASET_SOURCE == "A1":
    ENCODER_PATH = "models/label_encoder.pkl"
    models_dict = {
        "Random Forest": "models/random_forest.pkl",
        "XGBoost": "models/xgboost.pkl",
        "LightGBM": "models/lightgbm.pkl",
        "CatBoost": "models/catboost.pkl"
    }
else:
    ENCODER_PATH = "models/label_encoder_B1.pkl"
    models_dict = {
        "Random Forest": "models/random_forest_B1.pkl",
        "XGBoost": "models/xgboost_B1.pkl",
        "LightGBM": "models/lightgbm_B1.pkl",
        "CatBoost": "models/catboost_B1.pkl"
    }

encoder = joblib.load(ENCODER_PATH)

print("=" * 70)
print(f"SINGLE SAMPLE INFERENCE TEST (Dataset Source: {DATASET_SOURCE}, Index: {SAMPLE_INDEX})")
print("=" * 70)

if DATASET_SOURCE == "A1":
    # Load processed training features to see the schema
    features_schema = pd.read_csv("feature_engineering/output/Features_A.csv", nrows=0).columns
    
    # Load the clean dataset A1
    df = pd.read_csv("preprocessing/cleaned_dataset/clean_dataset_A.csv")
    sample = df.iloc[SAMPLE_INDEX]
    
    actual_label_name = sample["label"]
    
    # Separate features
    sample_features = sample.drop("label")
    X_sample = pd.DataFrame([sample_features])[features_schema]

else:  # Dataset B1
    # Load mapped features of B1
    df = pd.read_csv("dataset_B/output/mapped_features_B.csv")
    sample = df.iloc[SAMPLE_INDEX]
    
    # Normalise raw labels in B1 to match training
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
print(f"Actual Label             : {actual_label_name.upper()}")
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
    
    # Decode numeric prediction to actual label name
    pred_label_name = encoder.inverse_transform([pred_encoded])[0]
    
    # Check if prediction is correct
    status = "CORRECT" if pred_label_name.lower() == actual_label_name.lower() else "WRONG"
    
    print(f"{name:<20} : Predicted = {pred_label_name.upper():<12} | Status = {status}")

print("-" * 50)