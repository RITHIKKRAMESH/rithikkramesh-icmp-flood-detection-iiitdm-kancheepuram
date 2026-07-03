import os
import pandas as pd

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = "dataset/Dataset_A1.csv"

# ==========================================================

print("=" * 70)
print("MULTICLASS CLASS DISTRIBUTION")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

os.makedirs("reports", exist_ok=True)

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

distribution = df["label"].value_counts()

print("\nLabel Distribution\n")

print(distribution)

# ==========================================================
# PERCENTAGE
# ==========================================================

percentage = round((distribution / len(df)) * 100, 2)

print("\nAttack Percentage\n")

for label, count in distribution.items():

    print(f"{label:<15} {count:>8} ({percentage[label]}%)")

# ==========================================================
# IMBALANCE RATIO
# ==========================================================

majority = distribution.max()
minority = distribution.min()

print("\n" + "=" * 70)
print("DATASET IMBALANCE")
print("=" * 70)

print(f"Majority Class : {distribution.idxmax()} ({majority})")

print(f"Minority Class : {distribution.idxmin()} ({minority})")

print(f"\nImbalance Ratio = {round(majority/minority,2)} : 1")

# ==========================================================
# SAVE REPORT
# ==========================================================

report = pd.DataFrame({

    "Label": distribution.index,

    "Count": distribution.values,

    "Percentage": percentage.values

})

report.to_csv(

    "reports/class_distribution.csv",

    index=False

)

print("\nReport Saved Successfully")

print("reports/class_distribution.csv")

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION COMPLETED")
print("=" * 70)