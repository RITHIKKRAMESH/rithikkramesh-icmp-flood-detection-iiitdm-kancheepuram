import os
import numpy as np
import pandas as pd

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = "dataset/Dataset_A1.csv"

# ==========================================================

print("=" * 70)
print("MULTICLASS DATA PREPROCESSING")
print("=" * 70)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(DATASET_PATH)

print("\nOriginal Shape :", df.shape)
# ==========================================================
# MERGE NORMAL TRAFFIC CLASSES
# ==========================================================

print("\nConverting Normal Traffic Labels...")

label_column = "label"      # Change this if your label column has a different name

df[label_column] = df[label_column].replace({

    "NORMAL_ICMP": "Benign",

    "NORMAL_TCP": "Benign",

    "NORMAL_UDP": "Benign"

})

print("\nUpdated Class Distribution")

print(df[label_column].value_counts())

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
# REMOVE UNNECESSARY COLUMNS
# ==========================================================

remove_columns = []

for column in df.columns:

    if column.lower().startswith("unnamed"):

        remove_columns.append(column)

if len(remove_columns) > 0:

    df.drop(columns=remove_columns, inplace=True)

print("\nRemoved Columns")

print(remove_columns)

# ==========================================================
# KEEP ONLY MULTICLASS LABEL
# ==========================================================

if "Label_binary" in df.columns:

    df.drop(columns=["Label_binary"], inplace=True)

    print("\nLabel_binary Removed")

print("Target Column : Label_multi")

# ==========================================================
# OPTIMIZE DATA TYPES
# ==========================================================

float_columns = df.select_dtypes(include=["float64"]).columns

for col in float_columns:

    df[col] = df[col].astype("float32")

int_columns = df.select_dtypes(include=["int64"]).columns

for col in int_columns:

    df[col] = df[col].astype("int32")

print("\nData Types Optimized")

# ==========================================================
# FINAL SHAPE
# ==========================================================

print("\nFinal Dataset Shape :", df.shape)

print("\nRemaining Columns :", len(df.columns))

# ==========================================================
# SAVE CLEAN DATASET
# ==========================================================

os.makedirs(
    "preprocessing/cleaned_dataset",
    exist_ok=True
)

OUTPUT_PATH = "preprocessing/cleaned_dataset/clean_dataset_A.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nClean Dataset Saved Successfully")

print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)