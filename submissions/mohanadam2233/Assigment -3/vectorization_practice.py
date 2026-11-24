import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Training corpus
docs = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]
new_texts = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]
reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]

def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def show_matrix(matrix, feature_names, row_names=None, decimals=None):
    arr = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    if decimals is not None:
        arr = np.round(arr.astype(float), decimals)
    print("Columns:", list(feature_names))
    for i, row in enumerate(arr):
        label = row_names[i] if row_names else f"Row {i+1}"
        print(f"{label}: {row.tolist()}")

def tokens_from_vectorizer(vectorizer, text):
    return vectorizer.build_analyzer()(text)

# 1. One-hot encoding
heading("TASK 1 — ONE-HOT ENCODING")
one_hot_vocab = ["cash", "class", "free", "lunch", "notes", "win"]
one_hot = {word: [int(word == item) for item in one_hot_vocab] for word in one_hot_vocab}
print("Vocabulary order:", one_hot_vocab)
print("One-hot vectors for each word in 'free cash':")
for word in ["free", "cash"]:
    print(f"{word}: {one_hot[word]}")
sentence_words = ["win", "free", "cash"]
print("One-hot vectors for 'win free cash':")
for word in sentence_words:
    print(f"{word}: {one_hot[word]}")
print("Number of output vectors:", len(sentence_words), "(one vector per input word/token)")

# 2. Bag of Words
heading("TASK 2 — BAG OF WORDS")
count_vec = CountVectorizer()
X_count = count_vec.fit_transform(docs)
terms = count_vec.get_feature_names_out()
print("Vocabulary (term -> column index):")
print(dict(zip(terms, range(len(terms)))))
print("Matrix shape:", X_count.shape)
print("Full document-term matrix:")
show_matrix(X_count, terms, [f"D{i}" for i in range(1, 6)])
d1_row = X_count.toarray()[0]
print("D1 cells with value 2:")
for term, value in zip(terms, d1_row):
    if value == 2:
        print(f"{term}: {value}")
pair = count_vec.transform(["win free phone", "phone free win"])
print("Vector for 'win free phone':")
show_matrix(pair[0], terms, ["win free phone"])
print("Vector for 'phone free win':")
show_matrix(pair[1], terms, ["phone free win"])
print("Are the two vectors identical?", bool(np.array_equal(pair.toarray()[0], pair.toarray()[1])))

# 3. Tokens vs vocabulary size
heading("TASK 3 — TOKENS VS VOCABULARY SIZE")
total_tokens = int(X_count.sum())
print("Total token occurrences (CountVectorizer tokenization):", total_tokens)
print("Vocabulary size:", len(terms))
doc_frequency = np.asarray((X_count > 0).sum(axis=0)).ravel()
print("Words appearing in more than one document:")
print([(term, int(df)) for term, df in zip(terms, doc_frequency) if df > 1])

# 4. Preprocessing inside vectorizer
heading("TASK 4 — PREPROCESSING INSIDE THE VECTORIZER")
variants = [
    ("Default", CountVectorizer()),
    ("lowercase=False", CountVectorizer(lowercase=False)),
    ('stop_words="english"', CountVectorizer(stop_words="english"))
]
for label, vec in variants:
    vec.fit([docs[0]])
    print(f"{label} vocabulary:", vec.get_feature_names_out().tolist())
print("D1 raw text:", docs[0])
print("Default lowercasing merges 'Win' and 'WIN' into 'win'; lowercase=False keeps case variants separate.")
print("Punctuation '!' is not a token under the default word token pattern; the one-character token 'a' is also excluded.")
stop_vec = CountVectorizer(stop_words="english")
stop_vec.fit(docs)
stop_vocab = stop_vec.get_feature_names_out().tolist()
print('D1-D5 vocabulary with stop_words="english":', stop_vocab)
print("Is 'not' in that vocabulary?", "not" in stop_vocab)
print("Stop words removed from D5 that would otherwise be tokenized:",
      [w for w in CountVectorizer().build_analyzer()(docs[4]) if w not in stop_vocab])

# 5. Limiting the vocabulary
heading("TASK 5 — LIMITING THE VOCABULARY")
min_df_vec = CountVectorizer(min_df=2)
min_df_vec.fit(docs)
max_features_vec = CountVectorizer(max_features=5)
max_features_vec.fit(docs)
print("Full vocabulary size:", len(terms))
print("min_df=2 vocabulary:", min_df_vec.get_feature_names_out().tolist())
print("min_df=2 size:", len(min_df_vec.get_feature_names_out()))
print("max_features=5 vocabulary:", max_features_vec.get_feature_names_out().tolist())
print("max_features=5 size:", len(max_features_vec.get_feature_names_out()))

# 6. N-grams
heading("TASK 6 — N-GRAMS")
for ngram_range in [(1, 1), (1, 2)]:
    vec = CountVectorizer(ngram_range=ngram_range)
    vec.fit([docs[4]])
    print(f"D5 ngram_range={ngram_range} vocabulary:")
    print(vec.get_feature_names_out().tolist())
for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    vec = CountVectorizer(ngram_range=ngram_range)
    vec.fit(docs)
    print(f"D1-D5 ngram_range={ngram_range}: vocabulary size = {len(vec.get_feature_names_out())}")

# 7. TF-IDF
heading("TASK 7 — TF-IDF")
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(docs)
tfidf_terms = tfidf_vec.get_feature_names_out()
print("TF-IDF matrix (rounded to 2 decimals):")
show_matrix(X_tfidf, tfidf_terms, [f"D{i}" for i in range(1, 6)], decimals=2)
count_lookup = {term: int(count) for term, count in zip(terms, X_count.toarray()[0])}
tfidf_lookup_d1 = {term: float(weight) for term, weight in zip(tfidf_terms, X_tfidf.toarray()[0])}
print("D1 — word | count | TF-IDF:")
for term in tfidf_terms:
    if count_lookup.get(term, 0):
        print(f"{term} | {count_lookup[term]} | {tfidf_lookup_d1[term]:.6f}")
d2_counts = X_count.toarray()[1]
d2_tfidf = X_tfidf.toarray()[1]
print("D2 — word | count | TF-IDF:")
for term, count, weight in zip(terms, d2_counts, [dict(zip(tfidf_terms, d2_tfidf)).get(t, 0.0) for t in terms]):
    if count:
        print(f"{term} | {int(count)} | {weight:.6f}")

# 8. TF-IDF by hand
heading("TASK 8 — TF-IDF BY HAND FOR D2")
N = len(docs)
d2_tokens = CountVectorizer().build_analyzer()(docs[1])
d2_token_count = len(d2_tokens)
print("Number of D2 tokens counted by CountVectorizer:", d2_token_count)
for word in ["cash", "today"]:
    count = d2_tokens.count(word)
    df = int(doc_frequency[np.where(terms == word)[0][0]])
    tf = count / d2_token_count
    idf = math.log(N / df)
    weight = tf * idf
    sklearn_weight = float(dict(zip(tfidf_terms, d2_tfidf))[word])
    print(f"{word}: count={count}, TF=count/total_D2_tokens={count}/{d2_token_count}={tf:.6f}, "
          f"df={df}, IDF=ln({N}/{df})={idf:.6f}, TF-IDF={weight:.6f}, "
          f"scikit-learn TF-IDF={sklearn_weight:.6f}")
print("Hand-calculation ranking (cash vs today):", "cash > today" if
      (d2_tokens.count("cash") / d2_token_count) * math.log(N / int(doc_frequency[np.where(terms == "cash")[0][0]])) >
      (d2_tokens.count("today") / d2_token_count) * math.log(N / int(doc_frequency[np.where(terms == "today")[0][0]])) else "cash <= today")

# 9. Sparse matrix
heading("TASK 9 — SPARSE MATRIX")
total_cells = int(X_count.shape[0] * X_count.shape[1])
nonzero_cells = int(X_count.nnz)
zero_cells = total_cells - nonzero_cells
print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", f"{(zero_cells / total_cells) * 100:.2f}%")

# 10. Fit and transform
heading("TASK 10 — FIT AND TRANSFORM")
train_vec = CountVectorizer()
X_train = train_vec.fit_transform(docs[:4])
train_vocab = set(train_vec.get_feature_names_out())
print("Training vocabulary (D1-D4):", sorted(train_vocab))
X_new = train_vec.transform(new_texts)
show_matrix(X_new, train_vec.get_feature_names_out(), ["N1", "N2"])
for label, text in zip(["N1", "N2"], new_texts):
    tokens = train_vec.build_analyzer()(text)
    oov = sorted(set(tokens) - train_vocab)
    print(f"{label} analyzed tokens:", tokens)
    print(f"{label} out-of-vocabulary words:", oov)
print("The vectorizer does not infer that 'prize' and 'prizes' are related; stemming or lemmatization could normalize forms.")

# 11. Data leakage
heading("TASK 11 — DATA LEAKAGE")
train_only_vec = CountVectorizer()
train_only_vec.fit(docs[:4])
all_data_vec = CountVectorizer()
all_data_vec.fit(docs[:4] + new_texts)
vocab_train = set(train_only_vec.get_feature_names_out())
vocab_all = set(all_data_vec.get_feature_names_out())
leaked = sorted(vocab_all - vocab_train)
print("Vocabulary fitted on D1-D4:", sorted(vocab_train))
print("Vocabulary fitted on D1-D4 + N1 + N2:", sorted(vocab_all))
print("Words introduced by test texts (leaked into fitted vocabulary):", leaked)

# 12. No meaning
heading("TASK 12 — NO MEANING: REVIEW VECTORS AND COSINE SIMILARITY")
review_vec = CountVectorizer()
X_reviews = review_vec.fit_transform(reviews)
review_terms = review_vec.get_feature_names_out()
print("Review vocabulary:", review_terms.tolist())
show_matrix(X_reviews, review_terms, ["R1", "R2", "R3"])
similarity = cosine_similarity(X_reviews)
print("Cosine similarity matrix (R1, R2, R3):")
print(np.round(similarity, 4))
print("R1 similarity to R2:", f"{similarity[0,1]:.4f}")
print("R1 similarity to R3:", f"{similarity[0,2]:.4f}")
print("Are R1 and R2 more similar than R1 and R3?", bool(similarity[0,1] > similarity[0,2]))
