import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# LABEL FILE PATH
# ==========================================================

LABEL_PATH = "multiclass_pipeline/feature_engineering/output/Labels_A.csv"

# ==========================================================

print("=" * 70)
print("LABEL ENCODING")
print("=" * 70)

# Load Labels
labels = pd.read_csv(LABEL_PATH)

# Detect the label column automatically
label_column = labels.columns[0]

print("\nDetected Label Column :", label_column)

# ==========================================================
# ENCODE LABELS
# ==========================================================

encoder = LabelEncoder()

labels[label_column] = encoder.fit_transform(labels[label_column])

# ==========================================================
# SHOW MAPPING
# ==========================================================

print("\nLabel Mapping")

for index, attack in enumerate(encoder.classes_):

    print(f"{index} ---> {attack}")

# ==========================================================
# SAVE ENCODED LABELS
# ==========================================================

labels.to_csv(
    "multiclass_pipeline/feature_engineering/output/Labels_Encoded.csv",
    index=False
)

# ==========================================================
# SAVE LABEL ENCODER
# ==========================================================

os.makedirs("multiclass_pipeline/models", exist_ok=True)

joblib.dump(

    encoder,

    "multiclass_pipeline/models/label_encoder.pkl"

)

print("\nEncoded Labels Saved")

print("multiclass_pipeline/feature_engineering/output/Labels_Encoded.csv")

print("\nLabel Encoder Saved")

print("multiclass_pipeline/models/label_encoder.pkl")

print("\n" + "=" * 70)
print("LABEL ENCODING COMPLETED")
print("=" * 70)