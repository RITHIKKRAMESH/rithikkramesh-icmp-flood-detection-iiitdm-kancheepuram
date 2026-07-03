import os
import numpy as np
import pandas as pd

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = "dataset/Dataset_A3.csv"

# ==========================================================

print("=" * 70)
print("MULTICLASS DATA PREPROCESSING (Dataset A3)")
print("=" * 70)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("\nOriginal Shape :", df.shape)

# ==========================================================
# ALIGN LABEL CLASSES
# ==========================================================

print("\nAligning Labels...")

label_column = "label"

df[label_column] = df[label_column].replace({
    "TCP_syn_flood_attack": "DDOS_TCP",
    "ICMP_flood_attack": "DDOS_ICMP",
    "UDP_flood_attack": "DDOS_UDP",
    "NORMAL": "benign"
})

print("\nUpdated Class Distribution:")
print(df[label_column].value_counts())

# ==========================================================
# MAP FEATURES (A3 -> Standard Packet Schema)
# ==========================================================

print("\nMapping Protocol column...")

df["protocol"] = df["protocol"].astype(str).str.strip().str.upper()
df["proto"] = df["protocol"].replace({
    "ICMP": 1,
    "TCP": 6,
    "UDP": 17
})
# Handle any other numeric protocol strings if any
df["proto"] = pd.to_numeric(df["proto"], errors='coerce').fillna(0).astype(int)

print("\nMapping TCP Flags...")

df["flag"] = df["flag"].fillna("").astype(str)

df["fin"] = df["flag"].str.contains("FIN", case=False).astype(int)
df["syn"] = df["flag"].str.contains("SYN", case=False).astype(int)
df["rst"] = df["flag"].str.contains("RST", case=False).astype(int)
df["psh"] = df["flag"].str.contains("PSH", case=False).astype(int)
df["ack"] = df["flag"].str.contains("ACK", case=False).astype(int)
df["urg"] = df["flag"].str.contains("URG", case=False).astype(int)

# Length column maps to total_length
df["total_length"] = df["length"]

# Select final aligned packet-header features + label
selected_cols = [
    'proto', 'total_length', 'src_port', 'dst_port',
    'fin', 'syn', 'rst', 'psh', 'ack', 'urg', 'label'
]

df = df[selected_cols].copy()

# ==========================================================
# REMOVE DUPLICATE ROWS
# ==========================================================

duplicate_rows = df.duplicated().sum()

print("\nDuplicate Rows :", duplicate_rows)

if duplicate_rows > 0:
    df.drop_duplicates(inplace=True)

print("Shape After Removing Duplicates :", df.shape)

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

missing_values = df.isnull().sum().sum()

print("\nMissing Values :", missing_values)

if missing_values > 0:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

print("Shape After Removing Missing Values :", df.shape)

# ==========================================================
# HANDLE INFINITE VALUES
# ==========================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)
infinite_values = df.isnull().sum().sum()

if infinite_values > 0:
    df.dropna(inplace=True)

print("\nInfinite Values Handled")
print("Current Shape :", df.shape)

# ==========================================================
# OPTIMIZE DATA TYPES & SHUFFLE
# ==========================================================

float_columns = df.select_dtypes(include=["float64"]).columns
for col in float_columns:
    df[col] = df[col].astype("float32")

int_columns = df.select_dtypes(include=["int64"]).columns
for col in int_columns:
    df[col] = df[col].astype("int32")

print("\nData Types Optimized")

# Shuffle dataset because it is sorted by class and has no temporal column
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================================
# SAVE CLEAN DATASET
# ==========================================================

os.makedirs("multiclass_pipeline/preprocessing/cleaned_dataset", exist_ok=True)
OUTPUT_PATH = "multiclass_pipeline/preprocessing/cleaned_dataset/clean_dataset_A.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("\nClean Dataset Saved Successfully")
print(OUTPUT_PATH)
print("Final Shape :", df.shape)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)