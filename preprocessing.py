import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 1 : Load Dataset
# -------------------------------
print("Loading dataset...")

df = pd.read_csv("dataset/Dataset_A3.csv")

print("Dataset Loaded Successfully")
print("Shape :", df.shape)

# -------------------------------
# STEP 2 : Remove Missing Values
# -------------------------------
print("\nRemoving missing values...")

df = df.dropna()

print("Shape after removing missing values:", df.shape)

# -------------------------------
# STEP 3 : Remove Unnecessary Columns
# -------------------------------

columns_to_drop = [c for c in ["Flow ID", "Src IP", "Dst IP", "Timestamp"] if c in df.columns]

if columns_to_drop:
    df = df.drop(columns=columns_to_drop)

print("\nColumns removed successfully.")

# -------------------------------
# STEP 4 : Convert Labels to Numbers
# -------------------------------

print("\nEncoding Labels...")

df["label"] = df["label"].map({
    "benign": 0,
    "DDOS_ICMP": 1,
    "DDOS_TCP": 1,
    "DDOS_UDP": 1
})

print(df["label"].value_counts())

# -------------------------------
# STEP 5 : Separate Features and Labels
# -------------------------------

X = df.drop("label", axis=1)

y = df["label"]

print("\nFeatures Shape :", X.shape)

print("Labels Shape :", y.shape)

# -------------------------------
# STEP 6 : Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)

print("\nPreprocessing Completed Successfully.")