import pandas as pd

df = pd.read_csv("dataset/Dataset_A3.csv")

# Remove missing values
df = df.dropna()

# Remove unnecessary columns if they exist
cols_to_drop = [c for c in ["Flow ID", "Src IP", "Dst IP", "Timestamp"] if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Encode labels
df["label"] = df["label"].map({
    "benign": 0,
    "DDOS_ICMP": 1,
    "DDOS_TCP": 1,
    "DDOS_UDP": 1
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
print(df["label"].value_counts())