import os
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

DATASET_B = "dataset/Dataset_B3.csv"
FEATURES_A = "feature_engineering/output/Features_A.csv"
OUTPUT = "dataset_B/output"

os.makedirs(OUTPUT, exist_ok=True)

print("="*70)
print("FEATURE MAPPING (Dataset B3 -> A3)")
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
# EXTRACT TCP FLAGS FROM B3 tcp_flag
# ==========================================================

print("\nExtracting TCP flags from tcp_flag column...")

df_B["tcp_flag"] = df_B["tcp_flag"].fillna(0).astype(int)

df_B["fin"] = ((df_B["tcp_flag"] & 1) > 0).astype(int)
df_B["syn"] = ((df_B["tcp_flag"] & 2) > 0).astype(int)
df_B["rst"] = ((df_B["tcp_flag"] & 4) > 0).astype(int)
df_B["psh"] = ((df_B["tcp_flag"] & 8) > 0).astype(int)
df_B["ack"] = ((df_B["tcp_flag"] & 16) > 0).astype(int)
df_B["urg"] = ((df_B["tcp_flag"] & 32) > 0).astype(int)

# Align label classes
if "label" in df_B.columns:
    df_B["label"] = df_B["label"].astype(str).str.strip().replace({
        "NORMAL": "benign",
        "ICMP_flood_attack": "DDOS_ICMP",
        "TCP_syn_flood_attack": "DDOS_TCP",
        "UDP_flood_attack": "DDOS_UDP"
    })

# Reindex and build the mapped set
mapped = df_B[columns_A].copy()
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