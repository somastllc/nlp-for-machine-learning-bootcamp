import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "clean_sms_spam_practice.csv"

# Load clean dataset
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# Separate features and labels
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)

print("\n=== TRAIN TEST SPLIT ===")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

# =====================================
# BUILD 5 CLASSIFICATION MODELS
# =====================================

models = {
    "Model 1 - Count + Naive Bayes": Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ]),

    "Model 2 - TF-IDF + Naive Bayes": Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ]),

    "Model 3 - TF-IDF + Logistic Regression": Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Model 4 - Balanced Logistic Regression": Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),

    "Model 5 - TF-IDF + Stop Words + Balanced LR": Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: Training completed!")


# =====================================
# EVALUATE ALL 5 MODELS
# =====================================

results = {}

for name, model in models.items():

    predictions = model.predict(X_test)

    print(f"\n{'='*50}")
    print(name)
    print("="*50)

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    results[name] = predictions

# =====================================
# CONFUSION MATRICES
# =====================================

print("\n=== MODEL 3 CONFUSION MATRIX ===")
cm3 = confusion_matrix(
    y_test,
    results["Model 3 - TF-IDF + Logistic Regression"],
    labels=["ham", "spam"]
)
print(cm3)

print("\n=== MODEL 4 CONFUSION MATRIX ===")
cm4 = confusion_matrix(
    y_test,
    results["Model 4 - Balanced Logistic Regression"],
    labels=["ham", "spam"]
)
print(cm4)

# =====================================
# MANUAL METRIC CALCULATIONS
# =====================================

TN, FP, FN, TP = cm4.ravel()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\n=== MANUAL METRICS - MODEL 4 ===")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("Accuracy:", round(accuracy, 4))


# =====================================
# INSPECT MODEL 4 FEATURE WEIGHTS
# =====================================

model4 = models["Model 4 - Balanced Logistic Regression"]

vectorizer = model4.named_steps["vectorizer"]
classifier = model4.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()
weights = classifier.coef_[0]

word_weights = pd.DataFrame({
    "word": feature_names,
    "weight": weights
})

print("\n=== TOP SPAM WORDS ===")
print(word_weights.sort_values(
    by="weight", ascending=False
).head(15))

print("\n=== TOP HAM WORDS ===")
print(word_weights.sort_values(
    by="weight", ascending=True
).head(15))

# =====================================
# MISSED SPAM AND FALSE ALARMS
# =====================================

model4_predictions = results[
    "Model 4 - Balanced Logistic Regression"
]

test_results = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": model4_predictions
})

# Missed Spam
missed_spam = test_results[
    (test_results["actual"] == "spam") &
    (test_results["predicted"] == "ham")
]

# False Alarms
false_alarms = test_results[
    (test_results["actual"] == "ham") &
    (test_results["predicted"] == "spam")
]

print("\n=== 5 MISSED SPAM ===")
print(missed_spam["text"].head(5).to_string(index=False))

print("\n=== 5 FALSE ALARMS ===")
print(false_alarms["text"].head(5).to_string(index=False))

# =====================================
# TEST 5 NEW SMS MESSAGES
# =====================================

new_messages = [
    "Congratulations! You won a free iPhone. Call now!",
    "Hey, are you coming to class today?",
    "URGENT! Claim your prize by clicking this link.",
    "Hi friend, I will meet you tomorrow.",
    "Free entry in our weekly competition. Reply now!"
]

print("\n=== NEW SMS PREDICTIONS ===")

for message in new_messages:
    prediction = model4.predict([message])[0]
    probability = model4.predict_proba([message])[0]

    print("\nMessage:", message)
    print("Prediction:", prediction)
    print("Spam Probability:", round(probability[1], 4))