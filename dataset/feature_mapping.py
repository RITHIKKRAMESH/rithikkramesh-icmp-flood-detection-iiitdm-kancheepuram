import os
import pandas as pd
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

DATASET_B = "dataset/Dataset_B1.csv"
FEATURES_A = "feature_engineering/output/Features_A.csv"
OUTPUT = "dataset_B/output"

os.makedirs(OUTPUT, exist_ok=True)

print("="*70)
print("FEATURE MAPPING (Dataset B1 -> A1)")
print("="*70)

# ==========================================================
# LOAD DATA
# ==========================================================

df_B = pd.read_csv(DATASET_B, low_memory=False)
features_A = pd.read_csv(FEATURES_A, nrows=0)
columns_A = list(features_A.columns)

print(f"\nDataset A Features : {len(columns_A)}")
print(f"Dataset B Features : {len(df_B.columns)-1}")

# ==========================================================
# MAP B1 FEATURES TO A1
# ==========================================================

print("\nApplying Custom Feature Mapping...")

# 1. Header Length
if "header_length" in df_B.columns:
    df_B["Header_Length"] = df_B["header_length"]

# 2. Protocol Type
if "proto" in df_B.columns:
    df_B["Protocol Type"] = df_B["proto"]

# 3. Protocol flags (ICMP, TCP, UDP)
if "proto" in df_B.columns:
    df_B["ICMP"] = (df_B["proto"] == 1).astype(int)
    df_B["TCP"] = (df_B["proto"] == 6).astype(int)
    df_B["UDP"] = (df_B["proto"] == 17).astype(int)

# 4. Extract TCP flags from tcp_flag
if "tcp_flag" in df_B.columns:
    # bitwise check
    df_B["fin_flag_number"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 1) > 0).astype(int)
    df_B["syn_flag_number"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 2) > 0).astype(int)
    df_B["rst_flag_number"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 4) > 0).astype(int)
    df_B["psh_flag_number"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 8) > 0).astype(int)
    df_B["ack_flag_number"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 16) > 0).astype(int)
    df_B["urg_count"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 32) > 0).astype(int)
    df_B["ece_flag_number"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 64) > 0).astype(int)
    df_B["cwr_flag_number"] = ((df_B["tcp_flag"].fillna(0).astype(int) & 128) > 0).astype(int)

# ==========================================================
# CREATE REMAINING MISSING FEATURES
# ==========================================================

common = list(set(columns_A) & set(df_B.columns))
missing = list(set(columns_A) - set(df_B.columns))

print(f"Mapped/Common Features: {len(common)}")
print(f"Missing Features (set to 0): {len(missing)}")

for feature in missing:
    df_B[feature] = 0

# ==========================================================
# KEEP ONLY DATASET A FEATURES
# ==========================================================

mapped = df_B[columns_A].copy()

# Include the label column for evaluations
if "label" in df_B.columns:
    mapped["label"] = df_B["label"]

print(f"\nMapped Shape: {mapped.shape}")

# ==========================================================
# SAVE
# ==========================================================

OUTPUT_FILE = os.path.join(OUTPUT, "mapped_features_B.csv")
mapped.to_csv(OUTPUT_FILE, index=False)

print("\nMapped Dataset Saved:")
print(OUTPUT_FILE)

print("\n"+"="*70)
print("FEATURE MAPPING COMPLETED")
print("="*70)