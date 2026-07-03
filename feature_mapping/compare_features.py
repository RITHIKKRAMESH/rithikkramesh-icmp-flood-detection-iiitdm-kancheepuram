import pandas as pd

print("=" * 70)
print("FEATURE COMPARISON")
print("=" * 70)

# -------------------------------------------------------
# Load Features
# -------------------------------------------------------

A = pd.read_csv("feature_engineering/output/features_A.csv")
B = pd.read_csv("feature_engineering/output/features_B.csv")

features_A = set(A.columns)
features_B = set(B.columns)

# -------------------------------------------------------

common = sorted(features_A.intersection(features_B))

only_A = sorted(features_A - features_B)

only_B = sorted(features_B - features_A)

# -------------------------------------------------------

print("\nDataset A Features :", len(features_A))
print("Dataset B Features :", len(features_B))

print("\nCommon Features :", len(common))

for feature in common:
    print(feature)

print("\n" + "=" * 70)
print("ONLY IN DATASET A")
print("=" * 70)

for feature in only_A:
    print(feature)

print("\n" + "=" * 70)
print("ONLY IN DATASET B")
print("=" * 70)

for feature in only_B:
    print(feature)