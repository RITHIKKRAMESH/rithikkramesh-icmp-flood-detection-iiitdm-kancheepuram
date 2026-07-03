import pandas as pd

df = pd.read_csv("dataset/ICMP_ATTACK_DATASET.csv")

# Remove missing values
df = df.dropna()

# Remove unnecessary columns
df = df.drop(columns=[
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp"
])

# Encode labels
df["Label"] = df["Label"].map({
    "NORMAL": 0,
    "DDOS": 1
})

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nMissing Values")
print(df.isnull().sum().sum())

print("\nData Types")
print(df.dtypes.value_counts())

print("\nLabel Distribution")
print(df["Label"].value_counts())