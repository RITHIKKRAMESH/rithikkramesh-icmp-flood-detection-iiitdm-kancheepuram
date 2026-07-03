import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Dataset_A3.csv")

# Basic information
print("=" * 50)
print("Dataset Shape:")
print(df.shape)

print("\n" + "=" * 50)
print("Column Names:")
print(df.columns.tolist())

print("\n" + "=" * 50)
print("First 5 Rows:")
print(df.head())

print("\n" + "=" * 50)
print("Missing Values:")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Data Types:")
print(df.dtypes)

print(df["label"].value_counts())