import re
import math
import numpy as np
from collections import Counter
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
    arr = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    if names is None:
        print(arr)
    else:
        print("features:", list(names))
        print(arr)

# 1. One-hot encoding
heading("TASK 1 — ONE-HOT ENCODING")
vocab = ["cash", "class", "free", "lunch", "notes", "win"]
index = {word: i for i, word in enumerate(vocab)}
def one_hot(word):
    vector = [0] * len(vocab)
    if word in index:
        vector[index[word]] = 1
    return vector
for word in ["free", "cash"]:
    print(f"{word}: {one_hot(word)}")
for word in ["win", "free", "cash"]:
    print(f"{word}: {one_hot(word)}")
print("Sentence 'win free cash' has 3 word vectors; each vector has 6 features.")

# 2. Bag of Words
heading("TASK 2 — BAG OF WORDS / DOCUMENT-TERM MATRIX")
count_vec = CountVectorizer()
X = count_vec.fit_transform(docs)
features = count_vec.get_feature_names_out()
print("Vocabulary:", list(features))
print("Shape:", X.shape)
show_matrix(X, features)
d1 = dict(zip(features, X.toarray()[0]))
print("D1 cells equal to 2:", {w: int(c) for w, c in d1.items() if c == 2})
pair = count_vec.transform(["win free phone", "phone free win"])
print("win free phone:", pair.toarray()[0])
print("phone free win:", pair.toarray()[1])
print("Are vectors equal?", bool(np.array_equal(pair.toarray()[0], pair.toarray()[1])))

# 3. Tokens and vocabulary
heading("TASK 3 — TOKENS VS VOCABULARY SIZE")
token_lists = [re.findall(r"(?u)\b\w\w+\b", d.lower()) for d in docs]
token_count = sum(map(len, token_lists))
df_counts = Counter()
for tokens in token_lists:
    df_counts.update(set(tokens))
print("Total token occurrences (CountVectorizer token rules):", token_count)
print("Vocabulary size:", len(features))
print("Words in more than one document:", sorted([w for w, n in df_counts.items() if n > 1]))
print("Document frequency:", dict(sorted(df_counts.items())))

# 4. Preprocessing
heading("TASK 4 — PREPROCESSING INSIDE THE VECTORIZER")
for label, vec in [
    ("default", CountVectorizer()),
    ("lowercase=False", CountVectorizer(lowercase=False)),
    ('stop_words="english"', CountVectorizer(stop_words="english"))
]:
    vec.fit([docs[0]])
    print(label, "vocabulary:", list(vec.get_feature_names_out()))
print("D1 raw text:", docs[0])
sw = CountVectorizer(stop_words="english")
sw.fit(docs)
print('stop_words="english" on D1-D5 vocabulary:', list(sw.get_feature_names_out()))
print("'not' retained?", "not" in sw.vocabulary_)

# 5. Vocabulary limits
heading("TASK 5 — LIMITING THE VOCABULARY")
min_vec = CountVectorizer(min_df=2).fit(docs)
max_vec = CountVectorizer(max_features=5).fit(docs)
print("Full vocabulary size:", len(features), list(features))
print("min_df=2 size:", len(min_vec.vocabulary_), list(min_vec.get_feature_names_out()))
print("max_features=5 size:", len(max_vec.vocabulary_), list(max_vec.get_feature_names_out()))

# 6. N-grams
heading("TASK 6 — N-GRAMS")
for ngram in [(1, 1), (1, 2)]:
    vec = CountVectorizer(ngram_range=ngram).fit([docs[4]])
    print(f"D5 ngram_range={ngram} vocabulary:", list(vec.get_feature_names_out()))
for ngram in [(1, 1), (1, 2), (1, 3)]:
    vec = CountVectorizer(ngram_range=ngram).fit(docs)
    print(f"Corpus ngram_range={ngram} vocabulary size:", len(vec.vocabulary_))

# 7. TF-IDF
heading("TASK 7 — TF-IDF")
tfidf_vec = TfidfVectorizer()
T = tfidf_vec.fit_transform(docs)
tf_features = tfidf_vec.get_feature_names_out()
print("Vocabulary:", list(tf_features))
print("TF-IDF matrix (rounded to 2 decimals):")
print(np.round(T.toarray(), 2))
counts = count_vec.transform(docs).toarray()
tf_index = {w: i for i, w in enumerate(tf_features)}
for doc_i, label in [(0, "D1"), (1, "D2")]:
    print(label, "word | count | TF-IDF")
    for word in tf_features:
        c = int(counts[doc_i, list(features).index(word)]) if word in features else 0
        weight = T.toarray()[doc_i, tf_index[word]]
        if c:
            print(f"{word} | {c} | {weight:.6f}")

# 8. TF-IDF by hand
heading("TASK 8 — TEXTBOOK TF-IDF BY HAND FOR D2")
N = len(docs)
for word in ["cash", "today"]:
    tf = int(counts[1, list(features).index(word)])
    df = int(df_counts[word])
    idf = math.log(N / df)
    score = tf * idf
    print(f"{word}: TF={tf}, df={df}, N={N}, IDF=ln({N}/{df})={idf:.6f}, TF-IDF={score:.6f}")
print("Textbook ranking: cash > today" if math.log(N/df_counts["cash"]) > math.log(N/df_counts["today"]) else "Textbook ranking: today >= cash")
print("sklearn D2 weights:", {w: round(float(T.toarray()[1, tf_index[w]]), 6) for w in ["cash", "today"]})

# 9. Sparsity
heading("TASK 9 — SPARSE MATRIX")
dense = X.toarray()
total = dense.size
nonzero = int(np.count_nonzero(dense))
zero = total - nonzero
print("Total cells:", total)
print("Non-zero cells:", nonzero)
print("Zero cells:", zero)
print("Percentage zeros:", f"{(zero / total) * 100:.2f}%")
print("Storage format:", X.getformat(), "(scikit-learn returns a sparse matrix)")

# 10. Fit and transform
heading("TASK 10 — FIT AND TRANSFORM / OOV")
train = docs[:4]
train_vec = CountVectorizer()
train_X = train_vec.fit_transform(train)
new_X = train_vec.transform(new_texts)
train_vocab = set(train_vec.get_feature_names_out())
print("Training vocabulary:", sorted(train_vocab))
for label, text, row in zip(["N1", "N2"], new_texts, new_X.toarray()):
    tokens = re.findall(r"(?u)\b\w\w+\b", text.lower())
    oov = sorted(set(tokens) - train_vocab)
    print(f"{label}: {text}")
    print("Vector:", row)
    print("OOV words:", oov)

# 11. Data leakage
heading("TASK 11 — DATA LEAKAGE")
v_train = CountVectorizer().fit(train)
v_leaked = CountVectorizer().fit(train + new_texts)
v1, v2 = set(v_train.get_feature_names_out()), set(v_leaked.get_feature_names_out())
print("Train-only vocabulary:", sorted(v1))
print("Train + test vocabulary:", sorted(v2))
print("Words leaked from test set:", sorted(v2 - v1))

# 12. No meaning
heading("TASK 12 — BAG OF WORDS DOES NOT CAPTURE MEANING")
review_vec = CountVectorizer()
R = review_vec.fit_transform(reviews)
print("Vocabulary:", list(review_vec.get_feature_names_out()))
print("Review vectors:")
show_matrix(R, review_vec.get_feature_names_out())
sim = cosine_similarity(R)
print("Cosine similarity matrix (rounded to 4 decimals):")
print(np.round(sim, 4))
print("Similarity R1-R2:", f"{sim[0,1]:.4f}")
print("Similarity R1-R3:", f"{sim[0,2]:.4f}")
print("R1 more similar to R2 than R3?", bool(sim[0,1] > sim[0,2]))
