# --------------------------------
# ASSIGNMENT 04
# DATA PROCESSING
# --------------------------------

import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "sms_spam.csv"
df = pd.read_csv(DATA_PATH, encoding="latin-1")

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
df.info()

# Remove unnecessary columns
df = df[["v1", "v2"]]

print("\nAfter Removing Unnecessary Columns:")
print(df.head())

print("\nNew Dataset Shape:")
print(df.shape)

df.columns = ["label", "text"]

print(df.head())
print(df.columns.tolist())

print("\nDuplicate Messages:")
print(df["text"].duplicated().sum())

# Remove duplicate messages
df = df.drop_duplicates(subset="text")

print("\nAfter Removing Duplicates:")
print("New Shape:", df.shape)

print("Remaining Duplicates:", df["text"].duplicated().sum())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check class distribution
print("\nLabel Distribution:")
print(df["label"].value_counts())

# Check label percentages
print("\nLabel Percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# Check empty messages
print("\nEmpty Messages:")
print((df["text"].str.strip() == "").sum())

# Check average message length
print("\nAverage Message Length:")
print(df["text"].str.split().str.len().mean())

OUTPUT_PATH = BASE_DIR / "dataset" / "clean_sms_spam_practice.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nClean Dataset Saved Successfully!")
print("Final Shape:", df.shape)
print("Saved To:", OUTPUT_PATH)