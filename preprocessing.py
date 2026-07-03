import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 1 : Load Dataset
# -------------------------------
print("Loading dataset...")

df = pd.read_csv("dataset/ICMP_ATTACK_DATASET.csv")

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

columns_to_drop = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp"
]

df = df.drop(columns=columns_to_drop)

print("\nColumns removed successfully.")

# -------------------------------
# STEP 4 : Convert Labels to Numbers
# -------------------------------

print("\nEncoding Labels...")

df["Label"] = df["Label"].map({
    "NORMAL": 0,
    "DDOS": 1
})

print(df["Label"].value_counts())

# -------------------------------
# STEP 5 : Separate Features and Labels
# -------------------------------

X = df.drop("Label", axis=1)

y = df["Label"]

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