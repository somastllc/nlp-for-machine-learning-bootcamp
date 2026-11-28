"""
Assignment 4: train and evaluate five classical SMS text classifiers.
Input: dataset/clean_sms_spam_practice.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATA = Path("dataset/clean_sms_spam_practice.csv")
RANDOM_STATE = 7
TEST_SIZE = 0.2

def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def main():
    heading("STEP 1 — LOAD CLEANED DATA")
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found. Run data_processing_practice.py first.")
    df = pd.read_csv(DATA)
    if not {"label", "message"}.issubset(df.columns):
        raise ValueError("Cleaned CSV must contain 'label' and 'message' columns.")
    X = df["message"].astype(str)
    y = df["label"].astype(str)
    print(f"Rows loaded: {len(df)}")
    print(f"Label counts:\n{y.value_counts().to_string()}")

    heading("STEP 2 — STRATIFIED TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    for name, labels in [("Training", y_train), ("Test", y_test)]:
        print(f"{name} rows: {len(labels)}")
        print(f"{name} spam count: {(labels == 'spam').sum()}")
        print(f"{name} spam share: {(labels == 'spam').mean():.4%}")

    models = {
        "1. CountVectorizer + MultinomialNB": Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB())
        ]),
        "2. TfidfVectorizer + MultinomialNB": Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", MultinomialNB())
        ]),
        "3. TF-IDF + LogisticRegression": Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "4. TF-IDF + balanced LogisticRegression": Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "5. TF-IDF English stop words + balanced LogisticRegression": Pipeline([
            ("vectorizer", TfidfVectorizer(stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
    }
    predictions, rows = {}, []
    heading("STEP 3 — TRAIN FIVE PIPELINES AND CLASSIFICATION REPORTS")
    for name, model in models.items():
        print(f"\n{name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        print(classification_report(y_test, pred, labels=["ham", "spam"], zero_division=0))
        report = classification_report(y_test, pred, labels=["ham", "spam"],
                                       output_dict=True, zero_division=0)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Spam precision": report["spam"]["precision"],
            "Spam recall": report["spam"]["recall"],
            "Spam F1": report["spam"]["f1-score"]
        })

    heading("STEP 4 — MODEL COMPARISON")
    comparison = pd.DataFrame(rows).set_index("Model")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    best = comparison["Spam recall"].idxmax()
    print(f"\nBest spam recall: {best} ({comparison.loc[best, 'Spam recall']:.4f})")
    print("Naive Bayes comparison:")
    print(f"Count NB spam F1: {comparison.iloc[0]['Spam F1']:.4f}")
    print(f"TF-IDF NB spam F1: {comparison.iloc[1]['Spam F1']:.4f}")

    heading("STEP 5 — CONFUSION MATRICES: MODELS 3 AND 4")
    name3, name4 = list(models.keys())[2], list(models.keys())[3]
    matrices = {}
    for name in [name3, name4]:
        cm = confusion_matrix(y_test, predictions[name], labels=["ham", "spam"])
        matrices[name] = cm
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{name} — rows=true [ham, spam], columns=predicted [ham, spam]")
        print(cm)
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    heading("STEP 6 — HAND CALCULATION FOR MODEL 4")
    tn, fp, fn, tp = matrices[name4].ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.6f}")
    print(f"Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.6f}")
    print(f"F1        = 2PR / (P + R) = 2*{precision:.6f}*{recall:.6f}/({precision:.6f}+{recall:.6f}) = {f1:.6f}")
    print(f"Classification report spam precision/recall/F1: "
          f"{comparison.loc[name4, 'Spam precision']:.6f} / "
          f"{comparison.loc[name4, 'Spam recall']:.6f} / "
          f"{comparison.loc[name4, 'Spam F1']:.6f}")

    heading("STEP 7 — EFFECT OF CLASS WEIGHTING")
    tn3, fp3, fn3, tp3 = matrices[name3].ravel()
    tn4, fp4, fn4, tp4 = matrices[name4].ravel()
    print(f"Additional spam messages caught by balanced model (TP increase): {tp4 - tp3}")
    print(f"Additional false alarms caused (FP increase): {fp4 - fp3}")
    print(f"Missed spam change (FN): {fn4 - fn3}")

    heading("STEP 8 — WHAT MODEL 4 LEARNED: LOGISTIC REGRESSION WEIGHTS")
    fitted = models[name4]
    vectorizer = fitted.named_steps["vectorizer"]
    classifier = fitted.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    spam_index = list(classifier.classes_).index("spam")
    weights = classifier.coef_[0] if classifier.coef_.shape[0] == 1 else classifier.coef_[spam_index]
    top_positive = np.argsort(weights)[-10:][::-1]
    top_negative = np.argsort(weights)[:10]
    print("10 largest positive weights (push toward spam):")
    for i in top_positive:
        print(f"{feature_names[i]:<24} {weights[i]: .6f}")
    print("\n10 most negative weights (push toward ham):")
    for i in top_negative:
        print(f"{feature_names[i]:<24} {weights[i]: .6f}")

    heading("STEP 9 — ERROR ANALYSIS FOR MODEL 4")
    test_frame = pd.DataFrame({"message": X_test, "actual": y_test, "predicted": predictions[name4]})
    missed = test_frame[(test_frame.actual == "spam") & (test_frame.predicted == "ham")]
    false_alarms = test_frame[(test_frame.actual == "ham") & (test_frame.predicted == "spam")]
    print("\nFive missed spam messages (false negatives):")
    if missed.empty: print("None.")
    else:
        for i, row in missed.head(5).iterrows():
            print(f"- {row.message!r}")
    print("\nFive false alarms (false positives):")
    if false_alarms.empty: print("None.")
    else:
        for i, row in false_alarms.head(5).iterrows():
            print(f"- {row.message!r}")

    heading("STEP 10 — PREDICT FIVE NEW MESSAGES")
    new_messages = [
        "URGENT! You have won a free prize. Call now to claim your reward.",
        "Congratulations! Claim your cash bonus today by clicking this link.",
        "FREE entry in our weekly draw. Text WIN to 80000 now.",
        "Hi, are we still meeting for lunch at 1 pm?",
        "I am on the bus now. I will reach your house in ten minutes."
    ]
    probabilities = fitted.predict_proba(new_messages)
    spam_col = list(classifier.classes_).index("spam")
    new_predictions = fitted.predict(new_messages)
    for message, pred, probs in zip(new_messages, new_predictions, probabilities):
        print(f"Message: {message}")
        print(f"Prediction: {pred}; spam probability: {probs[spam_col]:.4f}")
        print("-" * 50)

    heading("END OF CLASSIFICATION PRACTICE")
    print("All figures above were generated from this dataset and this fixed split.")

if __name__ == "__main__":
    main()
