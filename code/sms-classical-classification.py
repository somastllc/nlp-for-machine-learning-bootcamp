# Lesson 4 — Classical Classification
# Project 1: SMS spam detection with TF-IDF + Logistic Regression and Naive Bayes.
# Follows lessons/04-classical-classification.md step by step.
# Setup once: python code/download-nltk-data.py
# Run first:  python code/sms-data-processing.py   (creates dataset/clean_sms_spam.csv)
# Then run:   python code/sms-classical-classification.py

import re

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = "dataset/clean_sms_spam.csv"
RANDOM_STATE = 42

# --------------------------------
# 1) Load the clean dataset
# sms-data-processing.py already removed missing text, fixed the labels,
# decoded HTML entities, normalized spaces and removed duplicates.
# One row = one SMS message with a label: ham (normal) or spam.
# --------------------------------
df = pd.read_csv(DATA_PATH)

print("\n=== 1) LOAD THE CLEAN DATASET ===")
print("Rows:", len(df), "| Columns:", df.columns.tolist())
print("Missing text:", df["text"].isna().sum(), "| Duplicate text:", df["text"].duplicated().sum())
print("Median words per message:", df["text"].str.split().str.len().median())
print("Labels:", df["label"].value_counts().to_dict())
print("Spam share:", round((df["label"] == "spam").mean() * 100, 1), "%")
print("Example ham: ", df[df["label"] == "ham"]["text"].iloc[0])
print("Example spam:", df[df["label"] == "spam"]["text"].iloc[0])

# --------------------------------
# 2) Stratified train/test split
# 70% training, 30% test. stratify keeps the same spam share in both parts.
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=RANDOM_STATE, stratify=df["label"]
)

print("\n=== 2) STRATIFIED TRAIN/TEST SPLIT ===")
print("Training messages:", len(X_train), "| spam share:", round((y_train == "spam").mean() * 100, 1), "%")
print("Test messages:    ", len(X_test), "| spam share:", round((y_test == "spam").mean() * 100, 1), "%")

# --------------------------------
# 3) Pipelines: TF-IDF + classifier
# A Pipeline chains the vectorizer and the model. fit() fits TF-IDF on the
# training text only, then trains the model; predict() reuses the same vocabulary.
# --------------------------------
logistic_model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
naive_bayes_model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("model", MultinomialNB()),
])
logistic_model.fit(X_train, y_train)
naive_bayes_model.fit(X_train, y_train)

print("\n=== 3) PIPELINES TRAINED ===")
print("Pipeline steps:", [name for name, step in logistic_model.steps])
print("Vocabulary size learned from training text:", len(logistic_model.named_steps["tfidf"].vocabulary_))

# --------------------------------
# 4) Logistic Regression results
# Positive class = spam, the class we want to catch.
# Precision: of messages flagged as spam, how many were spam?
# Recall: of all real spam, how many did we catch?
# Confusion matrix: rows = actual, columns = predicted.
# --------------------------------
logistic_predictions = logistic_model.predict(X_test)

print("\n=== 4) LOGISTIC REGRESSION RESULTS ===")
print("  Accuracy :", round(accuracy_score(y_test, logistic_predictions), 3))
print("  Precision:", round(precision_score(y_test, logistic_predictions, pos_label="spam"), 3), " (positive = spam)")
print("  Recall   :", round(recall_score(y_test, logistic_predictions, pos_label="spam"), 3), " (positive = spam)")
print("  F1-Score :", round(f1_score(y_test, logistic_predictions, pos_label="spam"), 3), " (positive = spam)")
print(pd.DataFrame(
    confusion_matrix(y_test, logistic_predictions, labels=["spam", "ham"]),
    index=["Actual: spam", "Actual: ham"],
    columns=["Pred: spam", "Pred: ham"],
))

# --------------------------------
# 5) Naive Bayes results
# Same split, same TF-IDF features, different classifier.
# --------------------------------
naive_bayes_predictions = naive_bayes_model.predict(X_test)

print("\n=== 5) NAIVE BAYES RESULTS ===")
print("  Accuracy :", round(accuracy_score(y_test, naive_bayes_predictions), 3))
print("  Precision:", round(precision_score(y_test, naive_bayes_predictions, pos_label="spam"), 3), " (positive = spam)")
print("  Recall   :", round(recall_score(y_test, naive_bayes_predictions, pos_label="spam"), 3), " (positive = spam)")
print("  F1-Score :", round(f1_score(y_test, naive_bayes_predictions, pos_label="spam"), 3), " (positive = spam)")
print(pd.DataFrame(
    confusion_matrix(y_test, naive_bayes_predictions, labels=["spam", "ham"]),
    index=["Actual: spam", "Actual: ham"],
    columns=["Pred: spam", "Pred: ham"],
))

# --------------------------------
# 6) Handling class imbalance: class_weight="balanced"
# Spam is the minority class, so errors on spam get more weight during training.
# --------------------------------
balanced_model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])
balanced_model.fit(X_train, y_train)
balanced_predictions = balanced_model.predict(X_test)

print("\n=== 6) LOGISTIC REGRESSION WITH class_weight='balanced' ===")
print("  Accuracy :", round(accuracy_score(y_test, balanced_predictions), 3))
print("  Precision:", round(precision_score(y_test, balanced_predictions, pos_label="spam"), 3), " (positive = spam)")
print("  Recall   :", round(recall_score(y_test, balanced_predictions, pos_label="spam"), 3), " (positive = spam)")
print("  F1-Score :", round(f1_score(y_test, balanced_predictions, pos_label="spam"), 3), " (positive = spam)")
print(pd.DataFrame(
    confusion_matrix(y_test, balanced_predictions, labels=["spam", "ham"]),
    index=["Actual: spam", "Actual: ham"],
    columns=["Pred: spam", "Pred: ham"],
))

# --------------------------------
# 7) Does preprocessing help? (Lesson 2 in action)
# Same balanced model, three versions of the text:
#   raw:          the text as it is
#   letters only: lowercase, delete everything except letters, tokenize, lemmatize
#   markers:      lowercase, REPLACE links, phone numbers and money with markers,
#                 tokenize, lemmatize
# Lesson 2: replacing keeps the fact that a number or link appeared; deleting loses it.
# --------------------------------
lemmatizer = WordNetLemmatizer()


def clean_letters_only(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(lemmatizer.lemmatize(token) for token in word_tokenize(text))


def clean_with_markers(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d{5,}", " phonenumber ", text)
    text = re.sub(r"£\s?\d+|\d+p\b", " money ", text)
    return " ".join(lemmatizer.lemmatize(token) for token in word_tokenize(text))


example = "Call 08712345678 to claim your £1000 prize! www.win.com"

print("\n=== 7) DOES PREPROCESSING HELP? (balanced Logistic Regression) ===")
print("Example message:", example)
print("  Letters only:", clean_letters_only(example))
print("  Markers:     ", clean_with_markers(example))

for name, train_text, test_text in [
    ("Raw text    ", X_train, X_test),
    ("Letters only", X_train.apply(clean_letters_only), X_test.apply(clean_letters_only)),
    ("Markers     ", X_train.apply(clean_with_markers), X_test.apply(clean_with_markers)),
]:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    model.fit(train_text, y_train)
    predictions = model.predict(test_text)
    print(" ", name,
          "| Precision:", round(precision_score(y_test, predictions, pos_label="spam"), 3),
          "| Recall:", round(recall_score(y_test, predictions, pos_label="spam"), 3),
          "| F1:", round(f1_score(y_test, predictions, pos_label="spam"), 3))

# --------------------------------
# 8) What the model learned
# Logistic Regression gives every word a weight. The largest positive weights
# push a message toward spam; the most negative weights push it toward ham.
# --------------------------------
feature_names = balanced_model.named_steps["tfidf"].get_feature_names_out()
weights = balanced_model.named_steps["model"].coef_[0]
order = weights.argsort()

print("\n=== 8) WHAT THE MODEL LEARNED (balanced Logistic Regression) ===")
print("Top 10 spam words:", feature_names[order[-10:][::-1]].tolist())
print("Top 10 ham words: ", feature_names[order[:10]].tolist())

# --------------------------------
# 9) Error analysis: read the mistakes
# Look at real messages the model got wrong and ask why.
# --------------------------------
results = pd.DataFrame({"text": X_test, "true": y_test, "predicted": balanced_predictions})
missed_spam = results[(results["true"] == "spam") & (results["predicted"] == "ham")]
false_alarms = results[(results["true"] == "ham") & (results["predicted"] == "spam")]

print("\n=== 9) ERROR ANALYSIS (balanced Logistic Regression) ===")
print("Missed spam:", len(missed_spam), "| False alarms:", len(false_alarms))
print("Three missed spam messages:")
for text in missed_spam["text"].head(3):
    print("  -", text)
print("Three false alarms:")
for text in false_alarms["text"].head(3):
    print("  -", text)

# --------------------------------
# 10) Single-message check and new messages
# Predict one real test message and compare with its true label: this spam
# message is missed by the first two models and caught by the balanced one.
# Then predict new messages with the spam probability (not a guarantee).
# --------------------------------
i = 64
one_message = X_test.iloc[i]

print("\n=== 10) SINGLE-MESSAGE CHECK ===")
print("  Message            :", one_message)
print("  Actual label       :", y_test.iloc[i])
print("  Logistic Regression:", logistic_model.predict([one_message])[0])
print("  Naive Bayes        :", naive_bayes_model.predict([one_message])[0])
print("  Balanced LR        :", balanced_model.predict([one_message])[0])

new_messages = [
    "URGENT! You have won a free reward, claim now",
    "Are we still meeting for lunch tomorrow?",
    "Call this number to get your prize today",
]

print("\n=== NEW MESSAGES (balanced Logistic Regression) ===")
for message in new_messages:
    label = balanced_model.predict([message])[0]
    spam_probability = balanced_model.predict_proba([message])[0][1]
    print(" ", label, "| spam probability:", round(spam_probability, 2), "|", message)