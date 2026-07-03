import pandas as pd

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = "dataset/Dataset_B3.csv"

# ==========================================================

print("=" * 70)
print("EXTERNAL DATASET (DATASET B) ANALYSIS")
print("=" * 70)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

# ==========================================================

print("\nDataset Shape")
print("-" * 70)
print(df.shape)

# ==========================================================

print("\nNumber of Rows")
print("-" * 70)
print(df.shape[0])

print("\nNumber of Columns")
print("-" * 70)
print(df.shape[1])

# ==========================================================

print("\nColumn Names")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):

    print(f"{i}. {column}")

# ==========================================================

print("\nData Types")
print("-" * 70)

print(df.dtypes)

# ==========================================================

print("\nMissing Values")
print("-" * 70)

missing = df.isnull().sum()

print(missing[missing > 0])

print("\nTotal Missing Values :", missing.sum())

# ==========================================================

print("\nDuplicate Rows")
print("-" * 70)

duplicates = df.duplicated().sum()

print(duplicates)

# ==========================================================

print("\nUnique Values Per Column")
print("-" * 70)

print(df.nunique())

# ==========================================================
# LABEL ANALYSIS
# ==========================================================

label_column = None

possible_labels = [

    "Label",
    "label",
    "Class",
    "class",
    "Attack",
    "attack",
    "Attack_Type",
    "Label_multi"

]

for column in df.columns:

    if column in possible_labels:

        label_column = column

        break

print("\nDetected Label Column")
print("-" * 70)

print(label_column)

if label_column is not None:

    print("\nAttack Distribution")
    print("-" * 70)

    print(df[label_column].value_counts())

# ==========================================================

print("\nDataset Information")
print("-" * 70)

df.info()

# ==========================================================

print("\nFirst Five Rows")
print("-" * 70)

print(df.head())

# ==========================================================

print("\nStatistical Summary")
print("-" * 70)

print(df.describe(include="all"))

print("\n" + "=" * 70)
print("DATASET B ANALYSIS COMPLETED")
print("=" * 70)