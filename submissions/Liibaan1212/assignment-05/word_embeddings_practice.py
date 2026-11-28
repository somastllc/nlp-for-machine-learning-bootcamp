"""Assignment 5: Word2Vec embeddings and SMS spam classification.

Input: dataset/clean_sms_spam.csv (or dataset/clean_sms_spam_practice.csv)
Run:   python word_embeddings_practice.py > output.txt
"""
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)
DATA_CANDIDATES = [
    Path("dataset/clean_sms_spam.csv"),
    Path("dataset/clean_sms_spam_practice.csv"),
    Path("clean_sms_spam.csv"),
    Path("clean_sms_spam_practice.csv"),
]
VECTOR_SIZE = 50
RANDOM_STATE = 7


def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def find_data():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cleaned SMS CSV not found. Put clean_sms_spam.csv or "
        "clean_sms_spam_practice.csv inside a dataset/ folder."
    )


def tokenize(text):
    """Lowercase and tokenize; remove punctuation-only tokens, keep numbers."""
    return [
        token.lower()
        for token in word_tokenize(str(text), preserve_line=True)
        if any(character.isalnum() for character in token)
    ]


def mean_vector(tokens, model):
    known_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not known_vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(known_vectors, axis=0).astype(np.float32)


def make_matrix(token_lists, model):
    return np.vstack([mean_vector(tokens, model) for tokens in token_lists])


def spam_metrics(y_true, predictions):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Spam precision": precision_score(y_true, predictions, pos_label="spam", zero_division=0),
        "Spam recall": recall_score(y_true, predictions, pos_label="spam", zero_division=0),
        "Spam F1": f1_score(y_true, predictions, pos_label="spam", zero_division=0),
        "False alarms": int(((np.asarray(y_true) == "ham") & (np.asarray(predictions) == "spam")).sum()),
    }


def main():
    # NLTK's current word_tokenize requires punkt_tab in recent versions.
    # preserve_line=True in word_tokenize uses NLTK tokenization without requiring Punkt data.

    heading("STEP 1 — EXPLORE THE CLEANED DATA")
    data_path = find_data()
    df = pd.read_csv(data_path)
    if not {"label", "message"}.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'label' and 'message'.")
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)
    print(f"Input file: {data_path}")
    print(f"Cleaned messages: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts().to_string())

    X = df["message"].reset_index(drop=True)
    y = df["label"].reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    for split_name, labels in [("Training", y_train), ("Test", y_test)]:
        print(f"{split_name} size: {len(labels)}")
        print(f"{split_name} spam count: {(labels == 'spam').sum()}")
        print(f"{split_name} spam share: {(labels == 'spam').mean():.4%}")
    print("Stratification keeps the ham/spam proportions approximately the same in both splits.")

    heading("STEP 2 — TOKENIZE THE SMS MESSAGES")
    sample_message = X_train.iloc[0]
    print("Before:", sample_message)
    print("After: ", tokenize(sample_message))
    print("Rule: convert tokens to lowercase; word_tokenize separates punctuation; punctuation-only tokens are removed.")
    print("Numbers and alphanumeric tokens are retained because they can be useful spam signals.")

    heading("STEP 3 — TRAIN CBOW AND SKIP-GRAM WORD2VEC")
    train_tokens = [tokenize(message) for message in X_train]
    test_tokens = [tokenize(message) for message in X_test]
    common_params = dict(vector_size=50, window=3, min_count=2, workers=1, seed=7, epochs=20)
    cbow = Word2Vec(sentences=train_tokens, sg=0, **common_params)
    skipgram = Word2Vec(sentences=train_tokens, sg=1, **common_params)
    print(f"CBOW vocabulary size: {len(cbow.wv)}")
    print(f"Skip-gram vocabulary size: {len(skipgram.wv)}")
    inspect_word = next((w for w in ["free", "call", "text", "you", "win"] if w in cbow.wv and w in skipgram.wv), None)
    if inspect_word is None:
        inspect_word = next(iter(cbow.wv.key_to_index))
    print(f"Chosen word: {inspect_word!r}")
    print("CBOW first five coordinates:", np.round(cbow.wv[inspect_word][:5], 5))
    print("Skip-gram first five coordinates:", np.round(skipgram.wv[inspect_word][:5], 5))

    heading("STEP 4 — INSPECT WORD EMBEDDINGS")
    print(f"Five nearest CBOW neighbors for {inspect_word!r}:")
    print(cbow.wv.most_similar(inspect_word, topn=5))
    print(f"Five nearest Skip-gram neighbors for {inspect_word!r}:")
    print(skipgram.wv.most_similar(inspect_word, topn=5))

    candidate_pairs = [("call", "free"), ("win", "prize"), ("text", "stop"), ("mobile", "phone")]
    pairs = [(a, b) for a, b in candidate_pairs if a in cbow.wv and b in cbow.wv and a in skipgram.wv and b in skipgram.wv][:2]
    if len(pairs) < 2:
        known = list(cbow.wv.key_to_index)
        for i in range(len(known) - 1):
            pair = (known[i], known[i + 1])
            if pair not in pairs and pair[0] in skipgram.wv and pair[1] in skipgram.wv:
                pairs.append(pair)
            if len(pairs) == 2:
                break
    for a, b in pairs:
        print(f"{a!r} vs {b!r} — CBOW cosine: {cbow.wv.similarity(a, b):.6f}; Skip-gram cosine: {skipgram.wv.similarity(a, b):.6f}")
    if pairs:
        a, b = pairs[0]
        va, vb = cbow.wv[a], cbow.wv[b]
        manual = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        library = float(cbow.wv.similarity(a, b))
        print(f"Manual cosine for {a!r}/{b!r} using CBOW:")
        print(f"dot product = {np.dot(va, vb):.8f}")
        print(f"length(A) = {np.linalg.norm(va):.8f}; length(B) = {np.linalg.norm(vb):.8f}")
        print(f"manual cosine = {manual:.8f}; model cosine = {library:.8f}; match = {np.isclose(manual, library)}")

    heading("STEP 5 — BUILD MEAN CBOW MESSAGE VECTORS")
    X_train_mean = make_matrix(train_tokens, cbow)
    X_test_mean = make_matrix(test_tokens, cbow)
    test_zero_count = int(np.all(X_test_mean == 0, axis=1).sum())
    print("Training mean-vector matrix shape:", X_train_mean.shape)
    print("Test mean-vector matrix shape:    ", X_test_mean.shape)
    print("Test messages represented by zero vectors:", test_zero_count)
    print("A zero vector is used when a message has no tokens in the fitted Word2Vec vocabulary.")

    heading("STEP 6 — TRAIN THE THREE CLASSIFIERS")
    models = {
        "Model 1 — Mean CBOW + Random Forest (200 trees)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=7, n_jobs=-1
        ),
        "Model 2 — Mean CBOW + Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=7
        ),
    }
    models["Model 3 — TF-IDF (1,2) + Logistic Regression"] = None
    predictions = {}

    models["Model 1 — Mean CBOW + Random Forest (200 trees)"].fit(X_train_mean, y_train)
    predictions["Model 1 — Mean CBOW + Random Forest (200 trees)"] = models["Model 1 — Mean CBOW + Random Forest (200 trees)"].predict(X_test_mean)

    models["Model 2 — Mean CBOW + Logistic Regression"].fit(X_train_mean, y_train)
    predictions["Model 2 — Mean CBOW + Logistic Regression"] = models["Model 2 — Mean CBOW + Logistic Regression"].predict(X_test_mean)

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    tfidf_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7)
    tfidf_lr.fit(X_train_tfidf, y_train)
    predictions["Model 3 — TF-IDF (1,2) + Logistic Regression"] = tfidf_lr.predict(X_test_tfidf)
    print("All three classifiers were trained on the same training split and evaluated on the same test split.")
    print("TF-IDF vocabulary size:", len(tfidf.vocabulary_))

    heading("STEP 7 — EVALUATE AND COMPARE")
    rows = []
    for name, pred in predictions.items():
        row = {"Model": name}
        row.update(spam_metrics(y_test, pred))
        rows.append(row)
    comparison = pd.DataFrame(rows).set_index("Model")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    for model_name in [
        "Model 2 — Mean CBOW + Logistic Regression",
        "Model 3 — TF-IDF (1,2) + Logistic Regression",
    ]:
        cm = confusion_matrix(y_test, predictions[model_name], labels=["ham", "spam"])
        print(f"\n{model_name}")
        print("Rows = actual [ham, spam]; columns = predicted [ham, spam]")
        print(pd.DataFrame(cm, index=["Actual ham", "Actual spam"], columns=["Predicted ham", "Predicted spam"]))
        tn, fp, fn, tp = cm.ravel()
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    heading("STEP 8 — ANALYZE MODEL 1 MISTAKES")
    model1_name = "Model 1 — Mean CBOW + Random Forest (200 trees)"
    pred1 = predictions[model1_name]
    test_frame = pd.DataFrame({"message": X_test.to_numpy(), "actual": y_test.to_numpy(), "predicted": pred1})
    missed_spam = test_frame[(test_frame.actual == "spam") & (test_frame.predicted == "ham")]
    false_alarms = test_frame[(test_frame.actual == "ham") & (test_frame.predicted == "spam")]
    print("Missed spam messages (up to five):")
    if missed_spam.empty:
        print("None — no missed spam messages on this test split.")
    else:
        for i, row in enumerate(missed_spam.head(5).itertuples(index=False), 1):
            print(f"{i}. {row.message}")
    print("False alarms (up to five):")
    if false_alarms.empty:
        print("None — no legitimate messages were incorrectly flagged on this test split.")
    else:
        for i, row in enumerate(false_alarms.head(5).itertuples(index=False), 1):
            print(f"{i}. {row.message}")

    heading("STEP 9 — INSPECT WHAT AVERAGING LOSES")
    message_a = "free prize call now"
    message_b = "now call prize free"
    vector_a = mean_vector(tokenize(message_a), cbow)
    vector_b = mean_vector(tokenize(message_b), cbow)
    print("Message A:", message_a)
    print("Message B:", message_b)
    print("Mean vector A:", np.round(vector_a, 5))
    print("Mean vector B:", np.round(vector_b, 5))
    print("Vectors match with np.allclose:", np.allclose(vector_a, vector_b))
    print("The mean representation ignores word order when the same known words occur with the same multiplicity.")

    heading("STEP 10 — PREDICT FIVE NEW MESSAGES")
    new_messages = [
        "Congratulations you have won a free cash prize call now",
        "URGENT claim your reward by texting WIN to 80000",
        "You have been selected for a free holiday offer reply YES",
        "Hey are we still meeting for lunch today",
        "Thanks for your help I will call you when I arrive",
    ]
    new_tokens = [tokenize(message) for message in new_messages]
    new_vectors = make_matrix(new_tokens, cbow)
    new_predictions = models[model1_name].predict(new_vectors)
    for message, prediction in zip(new_messages, new_predictions):
        print(f"Prediction: {prediction.upper():4s} | Message: {message}")

    heading("END OF ASSIGNMENT 5 PRACTICAL RUN")
    print("All results above were generated by this script from the selected cleaned dataset.")


if __name__ == "__main__":
    main()
