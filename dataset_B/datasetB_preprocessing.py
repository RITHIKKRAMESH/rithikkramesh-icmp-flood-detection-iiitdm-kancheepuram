import os
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

INPUT_PATH = "dataset_B/output/mapped_features_B.csv"

OUTPUT_FOLDER = "dataset_B/output"

OUTPUT_PATH = OUTPUT_FOLDER + "/processed_features_B.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================

print("=" * 70)
print("DATASET B PREPROCESSING")
print("=" * 70)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_PATH)

print("\nOriginal Shape :", df.shape)

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)

df.drop_duplicates(inplace=True)

print("Shape After Removing Duplicates :", df.shape)

# ==========================================================
# REPLACE INFINITE VALUES
# ==========================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==========================================================
# FILL MISSING VALUES
# ==========================================================

missing = df.isnull().sum().sum()

print("\nMissing Values :", missing)

df.fillna(0, inplace=True)

# ==========================================================
# CONVERT ALL FEATURES TO NUMERIC
# ==========================================================

for column in df.columns:

    df[column] = pd.to_numeric(df[column], errors="coerce")

df.fillna(0, inplace=True)

# ==========================================================
# FINAL CHECK
# ==========================================================

print("\nFinal Shape :", df.shape)

print("\nRemaining Missing Values :", df.isnull().sum().sum())

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(

    OUTPUT_PATH,

    index=False

)

print("\nProcessed Dataset Saved")

print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)