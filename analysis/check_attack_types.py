import pandas as pd

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = "dataset/Dataset_A3.csv"

# ==========================================================

print("=" * 70)
print("MULTICLASS ATTACK TYPE ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print("\nDataset Shape :", df.shape)

# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = ["label"]

print("\nChecking Required Columns...\n")

for column in required_columns:

    if column in df.columns:
        print(f"✔ {column} Found")
    else:
        print(f"✘ {column} Not Found")

# ==========================================================
# MULTICLASS DISTRIBUTION
# ==========================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

label_counts = df["label"].value_counts()

print(label_counts)

print("\nTotal Classes :", label_counts.shape[0])

# ==========================================================
# LABEL DISTRIBUTION SORTED
# ==========================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION (SORTED)")
print("=" * 70)

print(df["label"].value_counts().sort_index())

# ==========================================================
# UNIQUE LABELS
# ==========================================================

print("\n" + "=" * 70)
print("UNIQUE LABELS")
print("=" * 70)

labels = sorted(df["label"].dropna().unique())

for i, label in enumerate(labels, start=1):

    print(f"{i}. {label}")

print("\n" + "=" * 70)
print("LABEL ANALYSIS COMPLETED")
print("=" * 70)