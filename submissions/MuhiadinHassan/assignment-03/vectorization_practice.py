"""Reproducible scikit-learn and NumPy exercises for Assignment 3."""

import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DOCS = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]
NEW_TEXTS = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?",
]
REVIEWS = ["the food was tasty", "the food was delicious", "the food was terrible"]


def heading(title):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def matrix_text(matrix, feature_names):
    print("Feature order:", list(feature_names))
    print(matrix.toarray())


heading("1. ONE-HOT WORD ENCODING (CUSTOM PYTHON)")
one_hot_vocab = ["cash", "class", "free", "lunch", "notes", "win"]
one_hot_index = {word: i for i, word in enumerate(one_hot_vocab)}


def one_hot(word):
    vector = [0] * len(one_hot_vocab)
    vector[one_hot_index[word]] = 1
    return vector


print("Vocabulary/index order:", one_hot_vocab)
for word in ["free", "cash"]:
    print(f"{word} -> {one_hot(word)}")
for word in ["win", "free", "cash"]:
    print(f"Sentence token {word} -> {one_hot(word)}")
print("Vector count for 'win free cash':", len("win free cash".split()))


heading("2. COUNT BAG OF WORDS: D1-D5")
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(DOCS)
count_features = count_vectorizer.get_feature_names_out()
print("Vocabulary:", list(count_features))
print("Shape (documents, features):", count_matrix.shape)
matrix_text(count_matrix, count_features)
for query in ["win free phone", "phone free win"]:
    transformed = count_vectorizer.transform([query])
    print(f"Vector for {query!r}:", transformed.toarray()[0])


heading("3. TOKEN OCCURRENCES, VOCABULARY, AND DOCUMENT FREQUENCY")
analyzer = count_vectorizer.build_analyzer()
token_lists = [analyzer(document) for document in DOCS]
total_tokens = sum(len(tokens) for tokens in token_lists)
document_frequency = np.asarray((count_matrix > 0).sum(axis=0)).ravel()
repeated_documents = [
    (count_features[i], int(document_frequency[i]))
    for i in range(len(count_features))
    if document_frequency[i] > 1
]
print("Tokens per document (CountVectorizer analyzer):", [len(x) for x in token_lists])
print("Total token occurrences:", total_tokens)
print("Vocabulary size:", len(count_features))
print("Words appearing in more than one document (word, document frequency):", repeated_documents)


heading("4. PREPROCESSING OPTIONS INSIDE COUNT VECTORIZER")
for label, vectorizer in [
    ("Default", CountVectorizer()),
    ("lowercase=False", CountVectorizer(lowercase=False)),
    ("stop_words='english'", CountVectorizer(stop_words="english")),
]:
    vectorizer.fit([DOCS[0]])
    print(f"{label} vocabulary:", list(vectorizer.get_feature_names_out()))
default_d1_vocab = set(CountVectorizer().fit([DOCS[0]]).get_feature_names_out())
stop_d1_vocab = set(CountVectorizer(stop_words="english").fit([DOCS[0]]).get_feature_names_out())
print("D1 vocabulary terms removed by English stopword list:", sorted(default_d1_vocab - stop_d1_vocab))
print("Single-letter 'a' is a feature?", "a" in default_d1_vocab)
print("Exclamation mark is a feature?", "!" in default_d1_vocab)
print("Lowercase 'win' merged from Win/WIN by default?", "win" in default_d1_vocab)
english_vectorizer = CountVectorizer(stop_words="english")
english_matrix = english_vectorizer.fit_transform(DOCS)
english_vocab = list(english_vectorizer.get_feature_names_out())
print("D1-D5 vocabulary with English stopwords:", english_vocab)
print("Was 'not' retained?", "not" in english_vocab)
print("D5 tokens after this vectorizer's stopword filtering:", english_vectorizer.build_analyzer()(DOCS[4]))


heading("5. LIMITING VOCABULARY SIZE")
for label, vectorizer in [
    ("min_df=2", CountVectorizer(min_df=2)),
    ("max_features=5", CountVectorizer(max_features=5)),
]:
    vectorizer.fit(DOCS)
    features = list(vectorizer.get_feature_names_out())
    print(f"{label} vocabulary ({len(features)} terms):", features)
print("Full vocabulary size:", len(count_features))


heading("6. N-GRAMS")
for ngram_range in [(1, 1), (1, 2)]:
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit([DOCS[4]])
    print(f"D5 vocabulary for ngram_range={ngram_range}:", list(vectorizer.get_feature_names_out()))
for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(DOCS)
    print(f"D1-D5 vocabulary size for ngram_range={ngram_range}:", len(vectorizer.get_feature_names_out()))


heading("7. TF-IDF FOR D1-D5")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(DOCS)
tfidf_features = tfidf_vectorizer.get_feature_names_out()
print("TF-IDF matrix (rounded to 2 decimals), feature order:", list(tfidf_features))
print(np.round(tfidf_matrix.toarray(), 2))
for document_label, row_index in [("D1", 0), ("D2", 1)]:
    counts = count_matrix[row_index].toarray()[0]
    weights = tfidf_matrix[row_index].toarray()[0]
    print(f"{document_label} word/count/TF-IDF weight:")
    for word, count, weight in zip(count_features, counts, weights):
        if count:
            print(f"  {word}: count={int(count)}, tfidf={weight:.6f}")


heading("8. TEXTBOOK TF-IDF BY HAND FOR D2: CASH AND TODAY")
N = len(DOCS)
d2_tokens = analyzer(DOCS[1])
for word in ["cash", "today"]:
    tf = d2_tokens.count(word)
    col = count_vectorizer.vocabulary_[word]
    df = int(document_frequency[col])
    idf = float(np.log(N / df))
    tfidf = tf * idf
    sklearn_weight = float(tfidf_matrix[1, tfidf_vectorizer.vocabulary_[word]])
    print(f"{word}: raw_tf={tf}, N={N}, df={df}, idf=ln({N}/{df})={idf:.6f}, textbook_tf_idf={tfidf:.6f}, sklearn_l2_weight={sklearn_weight:.6f}")


heading("9. SPARSITY OF COUNT MATRIX")
total_cells = count_matrix.shape[0] * count_matrix.shape[1]
nonzero_cells = int(count_matrix.nnz)
zero_cells = total_cells - nonzero_cells
zero_percentage = 100 * zero_cells / total_cells if total_cells else 0.0
print("Shape:", count_matrix.shape)
print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print(f"Percentage of zeros: {zero_percentage:.2f}%")
print("Stored format:", count_matrix.getformat())


heading("10. FIT ON TRAINING DOCUMENTS, TRANSFORM NEW TEXTS")
train_vectorizer = CountVectorizer()
train_matrix = train_vectorizer.fit_transform(DOCS[:4])
train_vocab = set(train_vectorizer.get_feature_names_out())
print("Training vocabulary (D1-D4):", sorted(train_vocab))
for label, text in zip(["N1", "N2"], NEW_TEXTS):
    tokens = train_vectorizer.build_analyzer()(text)
    oov = sorted(set(tokens) - train_vocab)
    vector = train_vectorizer.transform([text])
    print(f"{label} text:", text)
    print(f"{label} tokens:", tokens)
    print(f"{label} vector:", vector.toarray()[0])
    print(f"{label} out-of-vocabulary words:", oov)


heading("11. DEMONSTRATING VOCABULARY LEAKAGE")
train_only_vectorizer = CountVectorizer().fit(DOCS[:4])
train_and_test_vectorizer = CountVectorizer().fit(DOCS[:4] + NEW_TEXTS)
train_only_vocab = set(train_only_vectorizer.get_feature_names_out())
combined_vocab = set(train_and_test_vectorizer.get_feature_names_out())
leaked = sorted(combined_vocab - train_only_vocab)
print("Vocabulary fitted on D1-D4 only:", sorted(train_only_vocab))
print("Vocabulary fitted on D1-D4 plus N1-N2:", sorted(combined_vocab))
print("Test-only words leaked into the fitted vocabulary:", leaked)


heading("12. BAG OF WORDS DOES NOT ENCODE MEANING: REVIEW COSINE SIMILARITY")
review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(REVIEWS)
review_features = review_vectorizer.get_feature_names_out()
print("Review vocabulary:", list(review_features))
print("Review vectors:")
matrix_text(review_matrix, review_features)
similarities = cosine_similarity(review_matrix)
print("Cosine similarity matrix (R1, R2, R3 order):")
print(np.round(similarities, 4))
print("R1-R2 similarity:", f"{similarities[0, 1]:.4f}")
print("R1-R3 similarity:", f"{similarities[0, 2]:.4f}")

