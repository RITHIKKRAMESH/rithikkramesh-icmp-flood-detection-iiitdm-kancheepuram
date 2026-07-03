import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=" * 70)
print("BINARY LABEL ENCODING MOCK")
print("=" * 70)

# Create a mock label encoder for binary classes: 0 -> benign, 1 -> attack
le = LabelEncoder()
le.fit(["benign", "attack"])

# Verify mapping
# Classes: ['attack', 'benign'] -> wait, sorted order is 'attack'=0, 'benign'=1.
# But we want 0 to be benign and 1 to be attack!
# So we can just set le.classes_ manually:
import numpy as np
le.classes_ = np.array(["benign", "attack"])

os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")

# Save a dummy Labels_Encoded.csv (matching Labels_A.csv since it's already 0/1)
labels_a = pd.read_csv("feature_engineering/output/Labels_A.csv")
labels_a.to_csv("feature_engineering/output/Labels_Encoded.csv", index=False)

print("\nMock Label Encoder Saved:")
print("0 ---> benign")
print("1 ---> attack")
print("\n" + "=" * 70)
print("LABEL ENCODING COMPLETED")
print("=" * 70)