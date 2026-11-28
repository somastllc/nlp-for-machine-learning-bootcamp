import pandas as pd

# 1. Load the original SMS dataset
file_path = "dataset/sms_spam_raw.csv"
df = pd.read_csv(file_path)

print("=== SMS DATASET LOADED ===")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
# 2. Check missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

missing_text = df["text"].isnull().sum()
print("\nMissing SMS messages:", missing_text)
# 3. Check duplicate SMS messages
print("\n=== DUPLICATE MESSAGES ===")

duplicate_count = df["text"].duplicated().sum()

print("Duplicate SMS messages:", duplicate_count)
# 4. Count messages in each label
print("\n=== LABEL DISTRIBUTION ===")
print(df["label"].value_counts(dropna=False))
# 5. Calculate spam percentage
print("\n=== SPAM PERCENTAGE ===")

spam_count = df["label"].str.strip().str.lower().str.replace(".", "", regex=False).eq("spam").sum()

total_messages = df["label"].notna().sum()

spam_percentage = (spam_count / total_messages) * 100

print("Total labeled messages:", total_messages)
print("Spam messages:", spam_count)
print(f"Spam percentage: {spam_percentage:.2f}%")
# 6. Clean the dataset
print("\n=== DATA CLEANING ===")

# Remove rows with missing SMS text
clean_df = df.dropna(subset=["text"]).copy()

# Remove extra spaces around messages
clean_df["text"] = clean_df["text"].str.strip()

# Remove empty messages
clean_df = clean_df[clean_df["text"] != ""]

# Standardize labels
clean_df["label"] = (
    clean_df["label"]
    .str.strip()
    .str.lower()
    .str.rstrip(".")
)

# Check duplicate messages after removing missing values
duplicates_before = clean_df["text"].duplicated().sum()

# Remove duplicate SMS messages
clean_df = clean_df.drop_duplicates(subset=["text"], keep="first")

print("Rows before cleaning:", len(df))
print("Rows after removing missing/empty messages:", len(df.dropna(subset=["text"])))
print("Duplicate messages removed:", duplicates_before)
print("Rows after cleaning:", len(clean_df))

print("\nCleaned label distribution:")
print(clean_df["label"].value_counts())
# 7. Save the cleaned dataset
output_path = "dataset/clean_sms_spam_practice.csv"

clean_df.to_csv(output_path, index=False)

print("\n=== CLEANED DATASET SAVED ===")
print("Saved to:", output_path)
print("Final dataset shape:", clean_df.shape)