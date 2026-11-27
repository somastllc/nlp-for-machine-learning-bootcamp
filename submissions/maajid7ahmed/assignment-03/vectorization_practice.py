"""
Assignment 3 – Vectorization practical
Run: python vectorization_practice.py > output.txt
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]
new_texts = ["Claim your free prize now", "Is the lunch meeting moved to tomorrow?"]
reviews = ["the food was tasty", "the food was delicious", "the food was terrible"]

def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def show_matrix(matrix, names=None):
    if names is not None:
        print("Features:", list(names))
    print(matrix.toarray() if hasattr(matrix, "toarray") else matrix)

# 1. One-hot encoding
heading("TASK 1 – ONE-HOT ENCODING")
vocab = ["cash", "class", "free", "lunch", "notes", "win"]
indices = {word: i for i, word in enumerate(vocab)}
def one_hot(word):
    vector = [0] * len(vocab)
    if word in indices:
        vector[indices[word]] = 1
    return vector
for word in ["free", "cash"]:
    print(f"{word}: {one_hot(word)}")
for word in ["win", "free", "cash"]:
    print(f"{word}: {one_hot(word)}")

# 2. Bag of Words
heading("TASK 2 – BAG OF WORDS / DOCUMENT-TERM MATRIX")
count_vec = CountVectorizer()
X = count_vec.fit_transform(docs)
features = count_vec.get_feature_names_out()
print("Vocabulary (alphabetical):", list(features))
print("Shape:", X.shape)
print("Document-term matrix (rows D1-D5; columns shown above):")
show_matrix(X)
d1 = X.toarray()[0]
print("D1 nonzero word counts:", {w: int(d1[i]) for i, w in enumerate(features) if d1[i]})
pair = count_vec.transform(["win free phone", "phone free win"])
print("Vector for 'win free phone':", pair.toarray()[0])
print("Vector for 'phone free win':", pair.toarray()[1])
print("Vectors equal:", bool(np.array_equal(pair.toarray()[0], pair.toarray()[1])))

# 3. Tokens and vocabulary size
heading("TASK 3 – TOKENS VS VOCABULARY SIZE")
analyzer = count_vec.build_analyzer()
token_lists = [analyzer(doc) for doc in docs]
print("Tokens per document:", [len(t) for t in token_lists])
print("Raw whitespace-separated word items:", sum(len(doc.split()) for doc in docs))
print("Token occurrences retained by CountVectorizer analyzer:", sum(map(len, token_lists)))
print("Vocabulary size:", len(features))
df = np.asarray((X > 0).sum(axis=0)).ravel()
print("Words appearing in more than one document:", [features[i] for i, n in enumerate(df) if n > 1])
print("Document frequency for shared words:", {features[i]: int(n) for i, n in enumerate(df) if n > 1})

# 4. Preprocessing
heading("TASK 4 – PREPROCESSING INSIDE THE VECTORIZER (D1 ONLY)")
for label, kwargs in [
    ("default", {}),
    ("lowercase=False", {"lowercase": False}),
    ('stop_words="english"', {"stop_words": "english"})
]:
    v = CountVectorizer(**kwargs).fit(docs[:1])
    print(f"{label} vocabulary:", list(v.get_feature_names_out()))
print("D1 raw text has one-letter 'a' and punctuation '!'; default token pattern drops one-letter tokens and punctuation.")
print("D1 stopword vocabulary removes:", sorted(set(CountVectorizer().fit(docs[:1]).get_feature_names_out()) -
                                               set(CountVectorizer(stop_words="english").fit(docs[:1]).get_feature_names_out())))
v_all_stop = CountVectorizer(stop_words="english").fit(docs)
print('D1-D5 stop_words="english" vocabulary:', list(v_all_stop.get_feature_names_out()))
print("'not' retained:", "not" in v_all_stop.vocabulary_)

# 5. Vocabulary limits
heading("TASK 5 – LIMITING THE VOCABULARY")
v_min = CountVectorizer(min_df=2).fit(docs)
v_max = CountVectorizer(max_features=5).fit(docs)
print("Full vocabulary size:", len(features), list(features))
print("min_df=2 vocabulary (size", len(v_min.vocabulary_), "):", list(v_min.get_feature_names_out()))
print("max_features=5 vocabulary (size", len(v_max.vocabulary_), "):", list(v_max.get_feature_names_out()))

# 6. N-grams
heading("TASK 6 – N-GRAMS")
for ngram in [(1, 1), (1, 2)]:
    v = CountVectorizer(ngram_range=ngram).fit(docs[4:5])
    print(f"D5 ngram_range={ngram} vocabulary:", list(v.get_feature_names_out()))
for ngram in [(1, 1), (1, 2), (1, 3)]:
    v = CountVectorizer(ngram_range=ngram).fit(docs)
    print(f"All documents ngram_range={ngram} vocabulary size:", len(v.vocabulary_))

# 7. TF-IDF
heading("TASK 7 – TF-IDF")
tfidf_vec = TfidfVectorizer()
T = tfidf_vec.fit_transform(docs)
tf_features = tfidf_vec.get_feature_names_out()
print("TF-IDF vocabulary:", list(tf_features))
print("TF-IDF matrix (rounded to 2 decimals):")
print(np.round(T.toarray(), 2))
for doc_index, label in [(0, "D1"), (1, "D2")]:
    counts = X.toarray()[doc_index]
    weights = T.toarray()[doc_index]
    print(f"{label} word | count | TF-IDF")
    for i, word in enumerate(features):
        if counts[i]:
            tf_i = tf_features.tolist().index(word)
            print(f"{word} | {int(counts[i])} | {weights[tf_i]:.6f}")

# 8. Hand TF-IDF
heading("TASK 8 – TEXTBOOK TF-IDF BY HAND FOR D2")
N = len(docs)
d2_tokens = analyzer(docs[1])
for word in ["cash", "today"]:
    tf = d2_tokens.count(word)
    document_frequency = int(df[np.where(features == word)[0][0]])
    idf = np.log(N / document_frequency)
    value = tf * idf
    print(f"{word}: TF={tf}, df={document_frequency}, N={N}, IDF=ln({N}/{document_frequency})={idf:.6f}, TF-IDF={value:.6f}")
print("Textbook ranking (larger weight first):", sorted(["cash", "today"], key=lambda w: np.log(N / int(df[np.where(features == w)[0][0]])), reverse=True))
print("Scikit-learn D2 ranking for these two:", sorted(["cash", "today"], key=lambda w: T.toarray()[1][tf_features.tolist().index(w)], reverse=True))

# 9. Sparsity
heading("TASK 9 – SPARSE MATRIX")
dense = X.toarray()
total_cells = dense.size
nonzero = int(np.count_nonzero(dense))
zeros = total_cells - nonzero
print("Total cells:", total_cells)
print("Non-zero cells:", nonzero)
print("Zero cells:", zeros)
print(f"Percentage zeros: {zeros / total_cells * 100:.2f}%")

# 10. Fit and transform
heading("TASK 10 – FIT AND TRANSFORM / OUT-OF-VOCABULARY")
train_vec = CountVectorizer()
train_matrix = train_vec.fit_transform(docs[:4])
print("Training vocabulary (D1-D4):", list(train_vec.get_feature_names_out()))
new_matrix = train_vec.transform(new_texts)
for label, text, row in zip(["N1", "N2"], new_texts, new_matrix.toarray()):
    analyzer_new = train_vec.build_analyzer()
    tokens = analyzer_new(text)
    oov = sorted(set(tokens) - set(train_vec.vocabulary_))
    print(f"{label} text: {text}")
    print(f"{label} vector:", row)
    print(f"{label} out-of-vocabulary words:", oov)

# 11. Data leakage
heading("TASK 11 – DATA LEAKAGE")
train_only = CountVectorizer().fit(docs[:4])
combined = CountVectorizer().fit(docs[:4] + new_texts)
v_train = set(train_only.get_feature_names_out())
v_combined = set(combined.get_feature_names_out())
print("Train-only vocabulary:", sorted(v_train))
print("Train + test vocabulary:", sorted(v_combined))
print("Words leaked from test set:", sorted(v_combined - v_train))

# 12. No meaning
heading("TASK 12 – BAG OF WORDS DOES NOT CAPTURE MEANING")
review_vec = CountVectorizer()
R = review_vec.fit_transform(reviews)
print("Review vocabulary:", list(review_vec.get_feature_names_out()))
print("Review vectors:")
print(R.toarray())
similarities = cosine_similarity(R)
print("Cosine similarity matrix (rounded to 4 decimals):")
print(np.round(similarities, 4))
print("Similarity R1-R2:", round(float(similarities[0, 1]), 4))
print("Similarity R1-R3:", round(float(similarities[0, 2]), 4))
