"""
Assignment 3 - Vectorization
Run:
    python vectorization_practice.py
To save the real terminal output:
    python vectorization_practice.py > output.txt
Dependencies: python -m pip install scikit-learn numpy nltk
"""

import math
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

DOCUMENTS = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]
TEST_DOCUMENTS = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?",
]
REVIEWS = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible",
]
DOC_IDS = ["D1", "D2", "D3", "D4", "D5"]


def heading(number, title):
    print("\n" + "=" * 78)
    print(f"TASK {number}: {title}")
    print("=" * 78)


def show_matrix(matrix, feature_names, row_names):
    print("Columns:", list(feature_names))
    print("Rows:")
    for row_name, row in zip(row_names, matrix.toarray()):
        print(f"{row_name}: {row.tolist()}")


def main():
    # Task 1
    heading(1, "Custom one-hot encoding")
    vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]
    one_hot = {
        word: [1 if word == vocab_word else 0 for vocab_word in vocabulary]
        for word in vocabulary
    }
    print("Vocabulary order:", vocabulary)
    for word in ["free", "cash", "win"]:
        print(f"Vector for '{word}': {one_hot[word]}")
    print("Sentence 'free cash':", [one_hot["free"], one_hot["cash"]])
    print("Sentence 'win free cash':",
          [one_hot["win"], one_hot["free"], one_hot["cash"]])
    print("Each word has one vector; a sentence has one vector per token.")

    # Task 2
    heading(2, "CountVectorizer and document-term matrix")
    count_vectorizer = CountVectorizer()
    dtm = count_vectorizer.fit_transform(DOCUMENTS)
    features = count_vectorizer.get_feature_names_out()
    show_matrix(dtm, features, DOC_IDS)
    d1 = dtm.toarray()[0]
    repeated = [(features[i], int(value)) for i, value in enumerate(d1) if value > 1]
    print("D1 terms with count greater than 1:", repeated)
    query_vectors = count_vectorizer.transform(["win free phone", "phone free win"])
    show_matrix(query_vectors, features, ["Q1: win free phone", "Q2: phone free win"])
    print("The two query rows are identical because bag-of-words ignores word order.")

    # Task 3
    heading(3, "Token count, vocabulary size, and document frequency")
    token_lists = [re.findall(r"\b\w+\b", doc.lower()) for doc in DOCUMENTS]
    total_tokens = sum(len(tokens) for tokens in token_lists)
    df = {}
    for tokens in token_lists:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    print("Total word occurrences (including one-character words):", total_tokens)
    print("CountVectorizer vocabulary size:", len(features))
    print("Words appearing in more than one document:",
          sorted([word for word, count in df.items() if count > 1]))
    print("Document frequencies:", dict(sorted(df.items())))

    # Task 4
    heading(4, "Lowercasing, punctuation, and stop words")
    d1_default = CountVectorizer()
    d1_lowercase_off = CountVectorizer(lowercase=False)
    d1_stopwords = CountVectorizer(stop_words="english")
    print("Default D1 vocabulary:", d1_default.fit(DOCUMENTS[:1]).get_feature_names_out().tolist())
    print("D1 vocabulary with lowercase=False:",
          d1_lowercase_off.fit(DOCUMENTS[:1]).get_feature_names_out().tolist())
    print("D1 vocabulary with English stop words:",
          d1_stopwords.fit(DOCUMENTS[:1]).get_feature_names_out().tolist())
    stop_vectorizer = CountVectorizer(stop_words="english")
    stop_matrix = stop_vectorizer.fit_transform(DOCUMENTS)
    stop_features = stop_vectorizer.get_feature_names_out().tolist()
    print("'not' in stop-word-filtered vocabulary:", "not" in stop_features)
    print("D5 tokens after English stop-word filtering:",
          stop_vectorizer.build_analyzer()(DOCUMENTS[4]))

    # Task 5
    heading(5, "Vocabulary limits: min_df and max_features")
    limited = CountVectorizer(min_df=2, max_features=5)
    limited.fit(DOCUMENTS)
    print("Full vocabulary size:", len(features))
    print("Limited vocabulary size:", len(limited.get_feature_names_out()))
    print("Limited vocabulary:", limited.get_feature_names_out().tolist())

    # Task 6
    heading(6, "Unigrams, bigrams, and trigrams")
    for ngram_range in [(1, 1), (1, 2)]:
        vec = CountVectorizer(ngram_range=ngram_range)
        vec.fit([DOCUMENTS[4]])
        print(f"D5 vocabulary for ngram_range={ngram_range}:")
        print(vec.get_feature_names_out().tolist())
    print("D5 bigram 'not good' is a useful opinion phrase.")
    for ngram_range in [(1, 1), (1, 2), (1, 3)]:
        vec = CountVectorizer(ngram_range=ngram_range)
        vec.fit(DOCUMENTS)
        print(f"Vocabulary size for ngram_range={ngram_range}: {len(vec.get_feature_names_out())}")

    # Task 7
    heading(7, "TF-IDF with scikit-learn")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(DOCUMENTS)
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    print("TF-IDF feature order:", tfidf_features.tolist())
    print("TF-IDF matrix (rounded to 2 decimals):")
    print(np.round(tfidf_matrix.toarray(), 2))
    print("D1 and D2: count and TF-IDF weight for each term in the document")
    count_array = dtm.toarray()
    for doc_index in [0, 1]:
        print(f"{DOC_IDS[doc_index]}:")
        for index, word in enumerate(features):
            count = int(count_array[doc_index, index])
            weight = float(tfidf_matrix[doc_index, list(tfidf_features).index(word)]) if word in tfidf_features else 0.0
            if count:
                print(f"  {word}: count={count}, tfidf={weight:.4f}")

    # Task 8
    heading(8, "Manual TF-IDF calculation for cash and today in D2")
    N = len(DOCUMENTS)
    for term in ["cash", "today"]:
        term_count = re.findall(r"\b\w+\b", DOCUMENTS[1].lower()).count(term)
        document_frequency = sum(term in set(re.findall(r"\b\w+\b", doc.lower())) for doc in DOCUMENTS)
        tf = term_count
        idf = math.log(N / document_frequency)
        manual_tfidf = tf * idf
        print(f"{term}: TF={tf}, N={N}, df={document_frequency}, "
              f"IDF=ln({N}/{document_frequency})={idf:.6f}, "
              f"TF-IDF={manual_tfidf:.6f}")
    print("Manual values use unsmoothed IDF and raw count TF; scikit-learn smooths IDF and L2-normalizes by default.")

    # Task 9
    heading(9, "Sparse matrix statistics")
    rows, columns = dtm.shape
    total_cells = rows * columns
    nonzero = int(dtm.nnz)
    zero = total_cells - nonzero
    print("DTM shape:", dtm.shape)
    print("Total cells:", total_cells)
    print("Non-zero cells:", nonzero)
    print("Zero cells:", zero)
    print(f"Percentage zeros: {zero / total_cells * 100:.2f}%")

    # Task 10
    heading(10, "Out-of-vocabulary test words")
    train_vectorizer = CountVectorizer()
    train_matrix = train_vectorizer.fit_transform(DOCUMENTS[:4])
    test_matrix = train_vectorizer.transform(TEST_DOCUMENTS)
    train_vocab = set(train_vectorizer.get_feature_names_out())
    show_matrix(test_matrix, train_vectorizer.get_feature_names_out(), ["N1", "N2"])
    for label, text in zip(["N1", "N2"], TEST_DOCUMENTS):
        tokens = train_vectorizer.build_analyzer()(text)
        oov = sorted(set(tokens) - train_vocab)
        print(f"{label} OOV words:", oov)
    stemmer = PorterStemmer()
    print("Porter stems: prize ->", stemmer.stem("prize"),
          "; prizes ->", stemmer.stem("prizes"))

    # Task 11
    heading(11, "Vocabulary leakage demonstration")
    train_only = CountVectorizer()
    train_only.fit(DOCUMENTS[:4])
    leaked_fit = CountVectorizer()
    leaked_fit.fit(DOCUMENTS[:4] + TEST_DOCUMENTS)
    train_only_vocab = set(train_only.get_feature_names_out())
    combined_vocab = set(leaked_fit.get_feature_names_out())
    print("Train-only vocabulary:", sorted(train_only_vocab))
    print("Vocabulary after incorrectly fitting on train + test:",
          sorted(combined_vocab))
    print("Words introduced from test data:", sorted(combined_vocab - train_only_vocab))
    print("Correct practice: fit preprocessing/vectorization on training data only, then transform test data.")

    # Task 12
    heading(12, "Cosine similarity of short reviews")
    review_vectorizer = CountVectorizer()
    review_matrix = review_vectorizer.fit_transform(REVIEWS)
    print("Review vocabulary:", review_vectorizer.get_feature_names_out().tolist())
    show_matrix(review_matrix, review_vectorizer.get_feature_names_out(), ["R1", "R2", "R3"])
    similarities = cosine_similarity(review_matrix)
    print("Cosine similarity matrix (rounded to 3 decimals):")
    print(np.round(similarities, 3))
    print(f"Similarity R1 tasty vs R2 delicious: {similarities[0, 1]:.3f}")
    print(f"Similarity R1 tasty vs R3 terrible: {similarities[0, 2]:.3f}")
    print("Count-based vectors do not understand that tasty and delicious are synonyms.")

    print("\n" + "=" * 78)
    print("END OF ASSIGNMENT 3 PRACTICAL")
    print("=" * 78)


if __name__ == "__main__":
    main()
