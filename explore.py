import pandas as pd

# Load the data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Basic info
print("=== TRAIN SET ===")
print(f"Shape: {train.shape}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(train.head())

print("\n=== TEST SET ===")
print(f"Shape: {test.shape}")

print("\n=== LABEL DISTRIBUTION (train) ===")
print(train.iloc[:, 2].value_counts())

print("\n=== MISSING VALUES ===")
print(train.isnull().sum())