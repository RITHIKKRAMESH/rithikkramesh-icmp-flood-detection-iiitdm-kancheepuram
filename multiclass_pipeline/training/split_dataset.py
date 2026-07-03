import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_PATH = os.path.join(
    PROJECT_ROOT,
    "feature_engineering",
    "output",
    "Features_A.csv"
)

LABEL_PATH = os.path.join(
    PROJECT_ROOT,
    "feature_engineering",
    "output",
    "Labels_A.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "training_data"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

# ==========================================================

print("=" * 70)
print("DATASET SPLITTING")
print("=" * 70)

# ==========================================================
# LOAD DATA
# ==========================================================

X = pd.read_csv(FEATURE_PATH)

y = pd.read_csv(LABEL_PATH)

# Convert dataframe into series
y = y.iloc[:, 0]

print("\nOriginal Dataset")

print("Features :", X.shape)
print("Labels   :", y.shape)

# ==========================================================
# LABEL ENCODING
# ==========================================================

print("\nEncoding Labels...")

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nLabel Mapping")

for i, label in enumerate(encoder.classes_):
    print(f"{label} --> {i}")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    encoder,
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

print("\nLabel Encoder Saved")

# ==========================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

train_end = int(len(X) * 0.70)
val_end = int(len(X) * 0.80)

X_train = X.iloc[:train_end]
y_train = y_encoded[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y_encoded[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y_encoded[val_end:]

# ==========================================================
# DISPLAY SHAPES
# ==========================================================

print("\nTraining Set :", X_train.shape)

print("Validation Set :", X_val.shape)

print("Testing Set :", X_test.shape)

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

print("\nTraining Class Distribution")

print(pd.Series(y_train).value_counts().sort_index())

print("\nValidation Class Distribution")

print(pd.Series(y_val).value_counts().sort_index())

print("\nTesting Class Distribution")

print(pd.Series(y_test).value_counts().sort_index())

# ==========================================================
# SAVE FILES
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train.to_csv(
    os.path.join(OUTPUT_DIR, "X_train.csv"),
    index=False
)

X_val.to_csv(
    os.path.join(OUTPUT_DIR, "X_val.csv"),
    index=False
)

X_test.to_csv(
    os.path.join(OUTPUT_DIR, "X_test.csv"),
    index=False
)

pd.DataFrame(
    y_train,
    columns=["Label"]
).to_csv(
    os.path.join(OUTPUT_DIR, "y_train.csv"),
    index=False
)

pd.DataFrame(
    y_val,
    columns=["Label"]
).to_csv(
    os.path.join(OUTPUT_DIR, "y_val.csv"),
    index=False
)

pd.DataFrame(
    y_test,
    columns=["Label"]
).to_csv(
    os.path.join(OUTPUT_DIR, "y_test.csv"),
    index=False
)

print("\nDatasets Saved Successfully")

print(os.path.join(OUTPUT_DIR, "X_train.csv"))
print(os.path.join(OUTPUT_DIR, "X_val.csv"))
print(os.path.join(OUTPUT_DIR, "X_test.csv"))
print(os.path.join(OUTPUT_DIR, "y_train.csv"))
print(os.path.join(OUTPUT_DIR, "y_val.csv"))
print(os.path.join(OUTPUT_DIR, "y_test.csv"))

print("\n" + "=" * 70)
print("DATASET SPLITTING COMPLETED")
print("=" * 70)