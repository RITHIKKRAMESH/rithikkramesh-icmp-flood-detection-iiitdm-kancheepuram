import os
import pandas as pd

# =====================================================
# DATASET PATH
# =====================================================

DATASET_PATH = "dataset/Dataset_A1.csv"

# =====================================================

print("=" * 70)
print("FEATURE ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

os.makedirs("reports", exist_ok=True)

# =====================================================
# FEATURE COUNT
# =====================================================

print("\nTotal Columns :", len(df.columns))

print("\nFeature Names\n")

for i, col in enumerate(df.columns, start=1):

    print(f"{i}. {col}")

# =====================================================
# NUMERICAL FEATURES
# =====================================================

numeric = df.select_dtypes(include=["int64", "float64"]).columns

print("\n" + "=" * 70)
print("NUMERICAL FEATURES")
print("=" * 70)

print("Total :", len(numeric))

# =====================================================
# CATEGORICAL FEATURES
# =====================================================

categorical = df.select_dtypes(include=["object"]).columns

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)

print("Total :", len(categorical))

for col in categorical:

    print(col)

# =====================================================
# CONSTANT FEATURES
# =====================================================

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

constant = []

for col in df.columns:

    if df[col].nunique() == 1:

        constant.append(col)

if len(constant) == 0:

    print("No Constant Features")

else:

    for col in constant:

        print(col)

# =====================================================
# UNIQUE VALUES
# =====================================================

print("\n" + "=" * 70)
print("UNIQUE VALUE COUNT")
print("=" * 70)

unique = pd.DataFrame({

    "Feature": df.columns,

    "Unique Values": df.nunique()

})

print(unique)

# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

summary = df.describe(include="all")

print(summary)

summary.to_csv(

    "reports/feature_summary.csv"

)

unique.to_csv(

    "reports/unique_values.csv",

    index=False

)

print("\nReports Saved Successfully")

print("reports/feature_summary.csv")

print("reports/unique_values.csv")

print("\n" + "=" * 70)
print("FEATURE ANALYSIS COMPLETED")
print("=" * 70)