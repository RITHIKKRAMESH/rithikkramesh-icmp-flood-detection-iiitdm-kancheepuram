import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/Dataset_A3.csv")

# -----------------------------
# Clean Dataset
# -----------------------------
df = df.dropna()

cols_to_drop = [c for c in ["Flow ID", "Src IP", "Dst IP", "Timestamp"] if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# -----------------------------
# Encode Labels
# -----------------------------
df["label"] = df["label"].map({
    "benign": 0,
    "DDOS_ICMP": 1,
    "DDOS_TCP": 1,
    "DDOS_UDP": 1
})

# -----------------------------
# Features and Labels
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "models/random_forest.pkl")

print("Model saved successfully!")