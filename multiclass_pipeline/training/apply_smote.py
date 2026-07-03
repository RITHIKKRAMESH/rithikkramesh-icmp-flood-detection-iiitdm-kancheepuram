import os
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

# ==========================================================
# DATA PATHS
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training", "training_data")
X_PATH = os.path.join(TRAINING_DATA_DIR, "X_train.csv")
Y_PATH = os.path.join(TRAINING_DATA_DIR, "y_train.csv")

# ==========================================================

print("=" * 70)
print("SMOTE BALANCING")
print("=" * 70)

# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

X_train = pd.read_csv(X_PATH)
y_train = pd.read_csv(Y_PATH)

# Convert DataFrame to Series
y_train = y_train.iloc[:, 0]

print("\nOriginal Training Set")

print("Features :", X_train.shape)
print("Labels   :", y_train.shape)

# ==========================================================
# ORIGINAL CLASS DISTRIBUTION
# ==========================================================

print("\nOriginal Class Distribution")
print("-" * 70)

original_distribution = Counter(y_train)

for cls, count in sorted(original_distribution.items()):
    print(f"Class {cls} : {count}")

# ==========================================================
# APPLY SMOTE
# ==========================================================

min_samples = min(original_distribution.values())

if min_samples > 1:
    k_neighbors = min(5, min_samples - 1)
    sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
    print(f"Using SMOTE (k_neighbors={k_neighbors})...")
else:
    from imblearn.over_sampling import RandomOverSampler
    sampler = RandomOverSampler(random_state=42)
    print("Using RandomOverSampler (due to class with 1 sample)...")

X_balanced, y_balanced = sampler.fit_resample(
    X_train,
    y_train
)

print("SMOTE Completed Successfully")

# ==========================================================
# BALANCED DISTRIBUTION
# ==========================================================

print("\nBalanced Class Distribution")
print("-" * 70)

balanced_distribution = Counter(y_balanced)

for cls, count in sorted(balanced_distribution.items()):
    print(f"Class {cls} : {count}")

# ==========================================================
# DISPLAY SHAPES
# ==========================================================

print("\nBalanced Dataset")

print("Features :", X_balanced.shape)
print("Labels   :", y_balanced.shape)

# ==========================================================
# SAVE BALANCED DATA
# ==========================================================

os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

pd.DataFrame(X_balanced).to_csv(
    os.path.join(TRAINING_DATA_DIR, "X_train_balanced.csv"),
    index=False
)

pd.DataFrame(y_balanced).to_csv(
    os.path.join(TRAINING_DATA_DIR, "y_train_balanced.csv"),
    index=False
)

print("\nBalanced Training Set Saved Successfully")

print(os.path.join(TRAINING_DATA_DIR, "X_train_balanced.csv"))
print(os.path.join(TRAINING_DATA_DIR, "y_train_balanced.csv"))

print("\n" + "=" * 70)
print("SMOTE BALANCING COMPLETED")
print("=" * 70)