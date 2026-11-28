# Lesson 5 — Project 2: SMS Spam Classification with Word Embeddings
# Run from the repository root after downloading the NLTK data and running
# code/sms-data-processing.py, as shown in lessons/05-word-embeddings.md.

import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = "dataset/clean_sms_spam.csv"
RANDOM_STATE = 42

# --------------------------------
# 1) Load the clean data and reproduce Project 1's split
# The cleaning script already removed missing text and duplicates. Using the
# same file, seed and stratification gives both projects the same test messages.
# Stratification keeps the spam proportions in the two sets close.
# --------------------------------
df = pd.read_csv(DATA_PATH)
train_text, test_text, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=RANDOM_STATE, stratify=df["label"]
)

print("\n=== 1) CLEAN DATA AND STRATIFIED SPLIT ===")
print("Training messages:", len(train_text), "| spam:", (train_labels == "spam").sum())
print("Test messages:    ", len(test_text), "| spam:", (test_labels == "spam").sum())

# --------------------------------
# 2) Tokenize and train Word2Vec
# Lowercase each message and use NLTK to separate punctuation from words.
# Fit on training text only. Even without labels, using test messages here
# would let their vocabulary and word relationships influence the model.
# --------------------------------
def tokenize(message):
    tokens = word_tokenize(message.lower())
    return [token for token in tokens if re.match(r"\w", token)]


train_tokens = [tokenize(message) for message in train_text]
test_tokens = [tokenize(message) for message in test_text]
word2vec = Word2Vec(
    train_tokens, vector_size=50, window=3, min_count=2, sg=0,
    epochs=20, workers=1, seed=RANDOM_STATE,
)
word_vectors = word2vec.wv

print("\n=== 2) TOKENIZE AND TRAIN WORD2VEC ===")
example_message = "Claim your FREE prize now!"
print("Example message:", example_message)
print("Lowercase and split:", example_message.lower().split())
print("NLTK with filtering:", tokenize(example_message))
print("Vocabulary size:", len(word_vectors), "| dimensions per word:", word2vec.vector_size)

# --------------------------------
# 3) Inspect the fitted word vectors
# Read a few coordinates, then use cosine similarity to compare vector
# directions. Nearest words reflect the contexts in this SMS corpus.
# --------------------------------
print("\n=== 3) INSPECT WORD VECTORS ===")
for word in ("free", "good"):
    if word in word_vectors:
        vector = word_vectors[word]
        print(word, "| first five coordinates:", vector[:5].round(3), "| nonzero values:", np.count_nonzero(vector))
        print("Nearest to", word, ":", [neighbor for neighbor, _ in word_vectors.most_similar(word, topn=5)])

for first, second in (("good", "great"), ("good", "free"), ("free", "win"), ("home", "house")):
    if first in word_vectors and second in word_vectors:
        print("Cosine similarity:", first, "vs", second, "=", round(float(word_vectors.similarity(first, second)), 2))

coverage_example = "the delivery was superb"
unknown_tokens = [token for token in tokenize(coverage_example) if token not in word_vectors]
print("Tokens without a vector in", repr(coverage_example), ":", unknown_tokens)

# --------------------------------
# 4) Average the word vectors to get one vector per SMS
# Messages have different lengths. Averaging each coordinate produces 50
# features for every message. For example, (2, 0, 4) and (0, 2, 2) average to
# (1, 1, 3). Each token occurrence gets equal weight, so a key word can get
# diluted. The mean also loses order: "good, not bad" and "bad, not good"
# produce the same features if they contain the same known tokens.
# Skip unknown tokens. With no known tokens, use an all-zero vector.
# --------------------------------
def average_vector(tokens, word_vectors):
    known_vectors = [word_vectors[token] for token in tokens if token in word_vectors]
    if known_vectors:
        return np.mean(known_vectors, axis=0)
    return np.zeros(word_vectors.vector_size, dtype=np.float32)


train_vectors = np.vstack([average_vector(tokens, word_vectors) for tokens in train_tokens])
test_vectors = np.vstack([average_vector(tokens, word_vectors) for tokens in test_tokens])

print("\n=== 4) AVERAGE INTO MESSAGE VECTORS ===")
print("Training matrix:", train_vectors.shape, "| test matrix:", test_vectors.shape)
print("First five values of the first training message:", train_vectors[0, :5].round(3))
no_known_words = sum(not any(token in word_vectors for token in tokens) for tokens in test_tokens)
print("Test messages with no known tokens:", no_known_words)

# --------------------------------
# 5) Train a Random Forest
# Each decision tree splits on numeric features. The forest combines the
# trees' predictions. Balanced class weights give spam, the smaller class,
# more weight during training.
# --------------------------------
classifier = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE,
)
classifier.fit(train_vectors, train_labels)
predictions = classifier.predict(test_vectors)

print("\n=== 5) TRAIN THE RANDOM FOREST ===")
print("Trees trained:", len(classifier.estimators_))

# --------------------------------
# 6) Evaluate on the held-out messages
# Spam is the positive class. Precision asks how many flagged messages really
# were spam. Recall asks how much of the actual spam we caught. F1 balances
# precision and recall. Accuracy counts correct predictions across both classes.
# --------------------------------
def scores(labels, predictions):
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Spam precision": precision_score(labels, predictions, pos_label="spam", zero_division=0),
        "Spam recall": recall_score(labels, predictions, pos_label="spam", zero_division=0),
        "Spam F1": f1_score(labels, predictions, pos_label="spam", zero_division=0),
    }


embedding_scores = scores(test_labels, predictions)
matrix = confusion_matrix(test_labels, predictions, labels=["spam", "ham"])

print("\n=== 6) EVALUATE THE RANDOM FOREST ===")
for metric, value in embedding_scores.items():
    print(metric, ":", round(value, 3))
print(pd.DataFrame(
    matrix, index=["Actual: spam", "Actual: ham"], columns=["Pred: spam", "Pred: ham"],
))
print("Missed spam:", matrix[0, 1], "| false alarms:", matrix[1, 0])

# --------------------------------
# 7) Predict new messages
# Use the same tokenizer and fitted vectors. An all-unknown message can
# still get a prediction, but its zero vector carries no word information.
# --------------------------------
messages = ["Are we still meeting tonight?", "WIN a free prize, claim now!"]
message_vectors = np.vstack([average_vector(tokenize(message), word_vectors) for message in messages])

print("\n=== 7) PREDICT NEW MESSAGES ===")
for message, label in zip(messages, classifier.predict(message_vectors)):
    print(message, "->", label)

# --------------------------------
# 8) Compare with Project 1
# Reuse the settings from Project 1, steps 4–6, on these exact same messages.
# One training-only TF-IDF fit supplies the identical features to all three
# baselines. This compares whole pipelines because both features and
# classifiers change. Use spam recall and false alarms to judge the trade-off.
# --------------------------------
tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
tfidf_train = tfidf.fit_transform(train_text)
tfidf_test = tfidf.transform(test_text)

print("\n=== 8) COMPARE WITH TF-IDF ===")
tfidf_example = tfidf.transform(["free prize"])
print("TF-IDF row for 'free prize':", tfidf_example.shape[1], "columns |", tfidf_example.nnz, "nonzero values")
good, great = tfidf.transform(["good", "great"])
if good.nnz and great.nnz:
    print("TF-IDF cosine similarity for 'good' vs 'great':", cosine_similarity(good, great)[0, 0])

baselines = {
    "TF-IDF + Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "TF-IDF + Naive Bayes": MultinomialNB(),
    "TF-IDF + balanced Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
}

comparison = {"Word2Vec + Random Forest": embedding_scores.copy()}
comparison["Word2Vec + Random Forest"]["False alarms"] = matrix[1, 0]
for name, model in baselines.items():
    model.fit(tfidf_train, train_labels)
    baseline_predictions = model.predict(tfidf_test)
    comparison[name] = scores(test_labels, baseline_predictions)
    baseline_matrix = confusion_matrix(test_labels, baseline_predictions, labels=["spam", "ham"])
    comparison[name]["False alarms"] = baseline_matrix[1, 0]

print("\nResults on the same test messages")
print(pd.DataFrame.from_dict(comparison, orient="index").round(3).to_string())
