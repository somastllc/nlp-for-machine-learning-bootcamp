import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load the cleaned SMS dataset
file_path = "dataset/clean_sms_spam_practice.csv"
df = pd.read_csv(file_path)

print("=== CLEANED SMS DATASET LOADED ===")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())
# 2. Separate features and labels
X = df["text"]
y = df["label"]

print("\n=== FEATURES AND LABELS ===")
print("Number of SMS messages (X):", len(X))
print("Number of labels (y):", len(y))

print("\nExample message:")
print(X.iloc[0])

print("\nCorrect label:")
print(y.iloc[0])
# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)

print("\n=== TRAINING AND TESTING SPLIT ===")
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())
# 4. Model 1: CountVectorizer + Multinomial Naive Bayes

model_1 = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model_1.fit(X_train, y_train)

predictions_1 = model_1.predict(X_test)

print("\n=== MODEL 1: COUNT VECTORIZER + NAIVE BAYES ===")
print("Accuracy:", accuracy_score(y_test, predictions_1))
print("\nClassification Report:")
print(classification_report(y_test, predictions_1))
# 5. Model 2: TF-IDF + Multinomial Naive Bayes

model_2 = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model_2.fit(X_train, y_train)

predictions_2 = model_2.predict(X_test)

print("\n=== MODEL 2: TF-IDF + NAIVE BAYES ===")
print("Accuracy:", accuracy_score(y_test, predictions_2))
print("\nClassification Report:")
print(classification_report(y_test, predictions_2))
# 6. Model 3: TF-IDF + Logistic Regression

model_3 = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model_3.fit(X_train, y_train)

predictions_3 = model_3.predict(X_test)

print("\n=== MODEL 3: TF-IDF + LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, predictions_3))
print("\nClassification Report:")
print(classification_report(y_test, predictions_3))
# 7. Model 4: TF-IDF + Balanced Logistic Regression

model_4 = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model_4.fit(X_train, y_train)

predictions_4 = model_4.predict(X_test)

print("\n=== MODEL 4: TF-IDF + BALANCED LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, predictions_4))
print("\nClassification Report:")
print(classification_report(y_test, predictions_4))
# 8. Model 5: TF-IDF + Balanced Logistic Regression + Stop Words

model_5 = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model_5.fit(X_train, y_train)

predictions_5 = model_5.predict(X_test)

print("\n=== MODEL 5: TF-IDF + BALANCED LOGISTIC REGRESSION + STOP WORDS ===")
print("Accuracy:", accuracy_score(y_test, predictions_5))
print("\nClassification Report:")
print(classification_report(y_test, predictions_5))
# 9. Compare all five models

results = {
    "Model": [
        "Count + NB",
        "TF-IDF + NB",
        "TF-IDF + LR",
        "Balanced LR",
        "Balanced LR + Stop Words"
    ],
    "Accuracy": [
        accuracy_score(y_test, predictions_1),
        accuracy_score(y_test, predictions_2),
        accuracy_score(y_test, predictions_3),
        accuracy_score(y_test, predictions_4),
        accuracy_score(y_test, predictions_5)
    ],
    "Spam Precision": [
        precision_score(y_test, predictions_1, pos_label="spam"),
        precision_score(y_test, predictions_2, pos_label="spam"),
        precision_score(y_test, predictions_3, pos_label="spam"),
        precision_score(y_test, predictions_4, pos_label="spam"),
        precision_score(y_test, predictions_5, pos_label="spam")
    ],
    "Spam Recall": [
        recall_score(y_test, predictions_1, pos_label="spam"),
        recall_score(y_test, predictions_2, pos_label="spam"),
        recall_score(y_test, predictions_3, pos_label="spam"),
        recall_score(y_test, predictions_4, pos_label="spam"),
        recall_score(y_test, predictions_5, pos_label="spam")
    ],
    "Spam F1": [
        f1_score(y_test, predictions_1, pos_label="spam"),
        f1_score(y_test, predictions_2, pos_label="spam"),
        f1_score(y_test, predictions_3, pos_label="spam"),
        f1_score(y_test, predictions_4, pos_label="spam"),
        f1_score(y_test, predictions_5, pos_label="spam")
    ]
}

comparison_df = pd.DataFrame(results)

print("\n=== COMPARISON OF ALL FIVE MODELS ===")
print(comparison_df.round(4).to_string(index=False))
# 10. Confusion matrices for Model 3 and Model 4

print("\n=== CONFUSION MATRIX: MODEL 3 ===")
cm_3 = confusion_matrix(
    y_test,
    predictions_3,
    labels=["ham", "spam"]
)
print(cm_3)

print("\n=== CONFUSION MATRIX: MODEL 4 ===")
cm_4 = confusion_matrix(
    y_test,
    predictions_4,
    labels=["ham", "spam"]
)
print(cm_4)
# 11. Manually calculate Model 4 metrics

tn, fp, fn, tp = cm_4.ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("\n=== MANUAL METRIC CALCULATIONS: MODEL 4 ===")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
# 12. Inspect the most important words learned by Model 4

vectorizer = model_4.named_steps["vectorizer"]
classifier = model_4.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

# Get the 10 words with the strongest positive weights
positive_indices = coefficients.argsort()[-10:][::-1]

# Get the 10 words with the strongest negative weights
negative_indices = coefficients.argsort()[:10]

print("\n=== TOP 10 WORDS ASSOCIATED WITH SPAM ===")
for index in positive_indices:
    print(f"{feature_names[index]}: {coefficients[index]:.4f}")

print("\n=== TOP 10 WORDS ASSOCIATED WITH HAM ===")
for index in negative_indices:
    print(f"{feature_names[index]}: {coefficients[index]:.4f}")
    # 13. Analyze incorrect predictions from Model 4

error_df = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "predicted": predictions_4
})

# Spam messages incorrectly predicted as ham
missed_spam = error_df[
    (error_df["actual"] == "spam") &
    (error_df["predicted"] == "ham")
]

# Ham messages incorrectly predicted as spam
false_alarms = error_df[
    (error_df["actual"] == "ham") &
    (error_df["predicted"] == "spam")
]

print("\n=== 5 MISSED SPAM MESSAGES ===")
print(missed_spam[["message", "actual", "predicted"]].head(5).to_string(index=False))

print("\n=== 5 FALSE ALARM MESSAGES ===")
print(false_alarms[["message", "actual", "predicted"]].head(5).to_string(index=False))
# 14. Test Model 4 using five original SMS messages

spam_samples = df[df["label"] == "spam"].sample(3, random_state=7)
ham_samples = df[df["label"] == "ham"].sample(2, random_state=7)

test_samples = pd.concat([spam_samples, ham_samples])

messages = test_samples["text"]
actual_labels = test_samples["label"]

predicted_labels = model_4.predict(messages)
probabilities = model_4.predict_proba(messages)

spam_index = list(model_4.classes_).index("spam")

print("\n=== TESTING FIVE ORIGINAL SMS MESSAGES ===")

for message, actual, predicted, probability in zip(
    messages,
    actual_labels,
    predicted_labels,
    probabilities[:, spam_index]
):
    print("\nMessage:", message)
    print("Actual label:", actual)
    print("Predicted label:", predicted)
    print(f"Spam probability: {probability:.4f}")