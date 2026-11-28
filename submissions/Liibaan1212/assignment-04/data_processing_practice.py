"""
Assignment 4: explore, clean and save the SMS spam dataset.
Expected input: dataset/sms_spam.csv
Expected output: dataset/clean_sms_spam_practice.csv
"""
from pathlib import Path
import pandas as pd

INPUT = Path("dataset/sms_spam.csv")
OUTPUT = Path("dataset/clean_sms_spam_practice.csv")

def find_columns(df):
    lower = {str(c).strip().lower(): c for c in df.columns}
    label_candidates = ["label", "v1", "category", "class", "target"]
    text_candidates = ["message", "text", "v2", "sms", "body"]
    label_col = next((lower[x] for x in label_candidates if x in lower), None)
    text_col = next((lower[x] for x in text_candidates if x in lower), None)
    if label_col is None or text_col is None:
        if len(df.columns) >= 2:
            label_col, text_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("CSV must contain a label column and a message/text column.")
    return label_col, text_col

def main():
    print("=" * 72)
    print("STEP 1 — LOAD AND EXPLORE SMS DATA")
    print("=" * 72)
    if not INPUT.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT}. Put sms_spam.csv in dataset/.")
    df = pd.read_csv(INPUT, encoding="utf-8", encoding_errors="replace")
    label_col, text_col = find_columns(df)
    print(f"Input file: {INPUT}")
    print(f"Original shape: {df.shape}")
    print(f"Detected label column: {label_col!r}")
    print(f"Detected text column: {text_col!r}")
    print(f"Rows: {len(df)}")
    print(f"Missing text rows: {df[text_col].isna().sum()}")
    print(f"Duplicate text rows (including repeated copies): {df[text_col].duplicated(keep='first').sum()}")
    print("\nOriginal label counts:")
    print(df[label_col].astype(str).str.strip().str.lower().value_counts(dropna=False).to_string())
    original_labels = df[label_col].astype(str).str.strip().str.lower().str.rstrip(".")
    original_spam_share = (original_labels == "spam").sum() / len(df)
    print(f"Original spam share (all rows): {original_spam_share:.4%}")

    print("\n" + "=" * 72)
    print("STEP 2 — STANDARDIZE AND CLEAN")
    print("=" * 72)
    clean = df[[label_col, text_col]].copy()
    clean.columns = ["label", "message"]
    clean["label"] = clean["label"].astype("string").str.strip().str.lower().str.rstrip(".")
    clean["message"] = clean["message"].astype("string").str.strip()
    clean = clean.dropna(subset=["label", "message"])
    clean = clean[(clean["label"] != "") & (clean["message"] != "")]
    clean["label"] = clean["label"].replace({
        "ham": "ham", "not spam": "ham", "legitimate": "ham", "0": "ham",
        "spam": "spam", "1": "spam"
    })
    unknown = sorted(set(clean["label"].dropna().unique()) - {"ham", "spam"})
    if unknown:
        raise ValueError(f"Unrecognized labels {unknown}. Expected ham/spam or 0/1.")
    before_duplicates = len(clean)
    clean = clean.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    print(f"Rows after dropping missing/blank text and labels: {before_duplicates}")
    print(f"Duplicate messages removed: {before_duplicates - len(clean)}")
    print(f"Clean rows: {len(clean)}")
    print("\nClean label counts:")
    print(clean["label"].value_counts().to_string())
    spam_share = (clean["label"] == "spam").mean()
    print(f"Spam share after cleaning: {spam_share:.4%}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"\nSaved cleaned dataset to: {OUTPUT}")
    print("=" * 72)

if __name__ == "__main__":
    main()
