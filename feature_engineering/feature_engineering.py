import os
import pandas as pd

# ==========================================================
# DATASET PATH
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "preprocessing", "cleaned_dataset", "clean_dataset_A.csv")

# ==========================================================

print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("\nOriginal Shape :", df.shape)

# ==========================================================
# TARGET COLUMN
# ==========================================================

TARGET_COLUMN = "label"

# ==========================================================
# REMOVE NON-TRAINING COLUMNS
# ==========================================================

REMOVE_COLUMNS = [

    "src_ip",
    "dst_ip",
    "timestamp"

]

available_remove = []

for column in REMOVE_COLUMNS:

    if column in df.columns:

        available_remove.append(column)

df.drop(columns=available_remove, inplace=True)

print("\nRemoved Columns")

print(available_remove)

# ==========================================================
# CREATE FEATURES AND LABELS
# ==========================================================

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

print("\nFeature Matrix Shape :", X.shape)

print("Label Shape :", y.shape)

print("\nTotal Training Features :", X.shape[1])

# ==========================================================
# SAVE
# ==========================================================

os.makedirs(

    os.path.join(PROJECT_ROOT, "feature_engineering", "output"),

    exist_ok=True

)

X.to_csv(

    os.path.join(PROJECT_ROOT, "feature_engineering", "output", "Features_A.csv"),

    index=False

)

y.to_csv(

    os.path.join(PROJECT_ROOT, "feature_engineering", "output", "Labels_A.csv"),

    index=False

)

print("\nFiles Saved")

print(os.path.join(PROJECT_ROOT, "feature_engineering", "output", "Features_A.csv"))

print(os.path.join(PROJECT_ROOT, "feature_engineering", "output", "Labels_A.csv"))

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)