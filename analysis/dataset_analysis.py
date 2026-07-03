import pandas as pd

# =====================================================
# DATASET PATH
# =====================================================

DATASET_PATH = "dataset/Dataset_A.csv"

# =====================================================

print("=" * 70)
print("DATASET ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\nDataset Shape")
print("-" * 40)

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

# =====================================================
# COLUMN NAMES
# =====================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# =====================================================
# DATA TYPES
# =====================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# =====================================================
# MISSING VALUES
# =====================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:

    print("No Missing Values Found")

else:

    print(missing)

# =====================================================
# DUPLICATE ROWS
# =====================================================

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicates = df.duplicated().sum()

print("Duplicate Rows :", duplicates)

# =====================================================
# MEMORY USAGE
# =====================================================

print("\n" + "=" * 70)
print("MEMORY USAGE")
print("=" * 70)

memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

print(f"{memory:.2f} MB")

# =====================================================
# NUMERICAL & CATEGORICAL FEATURES
# =====================================================

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

numeric = df.select_dtypes(include=["int64", "float64"]).columns

categorical = df.select_dtypes(include=["object"]).columns

print("\nNumerical Features :", len(numeric))
print("Categorical Features :", len(categorical))

print("\nCategorical Columns")

for col in categorical:
    print("-", col)

# =====================================================

print("\n" + "=" * 70)
print("DATASET ANALYSIS COMPLETED")
print("=" * 70)