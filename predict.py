import pandas as pd
import joblib

# -----------------------------
# Load Saved Model
# -----------------------------
model = joblib.load("models/icmp_ddos_model.pkl")

print("Model Loaded Successfully!\n")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/ICMP_ATTACK_DATASET.csv")

# -----------------------------
# Same Preprocessing
# -----------------------------
df = df.dropna()

df = df.drop(columns=[
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp"
])

df["Label"] = df["Label"].map({
    "NORMAL": 0,
    "DDOS": 1
})

# -----------------------------
# Select One Sample
# -----------------------------
sample = df.iloc[25]

actual_label = sample["Label"]

sample_features = sample.drop("Label")

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict([sample_features])[0]

# -----------------------------
# Display Result
# -----------------------------
print("="*50)

print("Actual Label      :", "DDOS" if actual_label == 1 else "NORMAL")

print("Predicted Label   :", "DDOS" if prediction == 1 else "NORMAL")

if prediction == actual_label:
    print("\nPrediction Status : CORRECT")
else:
    print("\nPrediction Status : WRONG")

print("="*50)