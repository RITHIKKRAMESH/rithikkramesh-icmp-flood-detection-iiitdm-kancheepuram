import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/ICMP_ATTACK_DATASET.csv")

# -----------------------------
# Clean Dataset
# -----------------------------
df = df.dropna()

df = df.drop(columns=[
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp"
])

# -----------------------------
# Encode Labels
# -----------------------------
df["Label"] = df["Label"].map({
    "NORMAL": 0,
    "DDOS": 1
})

# -----------------------------
# Features and Labels
# -----------------------------
X = df.drop("Label", axis=1)
y = df["Label"]

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
joblib.dump(model, "models/icmp_ddos_model.pkl")

print("Model saved successfully!")