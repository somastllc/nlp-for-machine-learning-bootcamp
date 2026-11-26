# SMS Data Processing
# Cleans dataset/sms_spam_raw.csv into dataset/clean_sms_spam.csv for the SMS projects.
# Run this first: python code/sms-data-processing.py

import html

import pandas as pd

RAW_PATH = "dataset/sms_spam_raw.csv"
CLEAN_PATH = "dataset/clean_sms_spam.csv"

# --------------------------------
# 1) Load + initial snapshot
# --------------------------------
df = pd.read_csv(RAW_PATH)

print("\n=== 1) INITIAL SNAPSHOT ===")
print("Rows:", len(df), "| Columns:", df.columns.tolist())
print(df.head())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== LABEL VALUES ===")
print(df["label"].value_counts())

# --------------------------------
# 2) Drop rows with missing text
# A message with no text cannot be classified, so we remove it.
# --------------------------------
before = len(df)
df = df.dropna(subset=["text"])

print("\n=== 2) DROP MISSING TEXT ===")
print("Rows:", before, "->", len(df))

# --------------------------------
# 3) Normalize the labels
# "Spam", "HAM ", "spam." and " ham" all mean the same two classes.
# strip() removes spaces, lower() fixes the case, rstrip(".") removes the dot.
# --------------------------------
df["label"] = df["label"].str.strip().str.lower().str.rstrip(".")

print("\n=== 3) NORMALIZE LABELS ===")
print(df["label"].value_counts())

# --------------------------------
# 4) Decode HTML entities
# Some messages contain web leftovers: "&lt;" means "<", "&gt;" means ">", "&amp;" means "&".
# html.unescape turns them back into the real characters.
# --------------------------------
has_entities = df["text"].str.contains("&lt;|&gt;|&amp;")
example_index = has_entities.idxmax()

print("\n=== 4) DECODE HTML ENTITIES ===")
print("Messages with HTML entities:", has_entities.sum())
print("Before:", df.loc[example_index, "text"])
df["text"] = df["text"].apply(html.unescape)
print("After: ", df.loc[example_index, "text"])

# --------------------------------
# 5) Normalize whitespace
# split() breaks on any run of spaces; " ".join() puts single spaces back,
# which also removes spaces at the start and end.
# --------------------------------
extra_spaces = (df["text"] != df["text"].str.split().str.join(" ")).sum()
df["text"] = df["text"].str.split().str.join(" ")

print("\n=== 5) NORMALIZE WHITESPACE ===")
print("Messages with extra spaces fixed:", extra_spaces)

# --------------------------------
# 6) Remove duplicates
# Done AFTER cleaning, so messages that differed only in spaces are caught too,
# and BEFORE any train/test split in the projects.
# --------------------------------
before = len(df)
df = df.drop_duplicates(subset=["text"])

print("\n=== 6) REMOVE DUPLICATES ===")
print("Rows:", before, "->", len(df))

# --------------------------------
# 7) Final snapshot + save
# --------------------------------
print("\n=== 7) FINAL SNAPSHOT ===")
print("Rows:", len(df))
print("Missing text:", df["text"].isna().sum())
print("Labels:", df["label"].value_counts().to_dict())
print("Spam share:", round((df["label"] == "spam").mean() * 100, 1), "%")

df.to_csv(CLEAN_PATH, index=False)
print("\nSaved", CLEAN_PATH)