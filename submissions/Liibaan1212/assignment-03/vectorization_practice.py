from collections import Counter
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

D = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]
N = ["Claim your free prize now", "Is the lunch meeting moved to tomorrow?"]
R = ["the food was tasty", "the food was delicious", "the food was terrible"]


def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_matrix(matrix, feature_names, row_names=None):
    print("Features:", list(feature_names))
    print("Shape:", matrix.shape)
    print(matrix.toarray())
    if row_names:
        for name, row in zip(row_names, matrix.toarray()):
            print(f"{name}: {row.tolist()}")


heading("TASK 1 — ONE-HOT ENCODING")
vocab = ["cash", "class", "free", "lunch", "notes", "win"]
index = {word: i for i, word in enumerate(vocab)}
def one_hot(word):
    v = [0] * len(vocab)
    if word in index:
        v[index[word]] = 1
    return v
for word in ["free", "cash"]:
    print(f"{word}: {one_hot(word)}")
for word in "win free cash".split():
    print(f"{word}: {one_hot(word)}")

heading("TASK 2 — BAG OF WORDS / DOCUMENT-TERM MATRIX")
cv = CountVectorizer()
X = cv.fit_transform(D)
terms = cv.get_feature_names_out()
show_matrix(X, terms, [f"D{i}" for i in range(1, 6)])
print("D1 nonzero cells:", [(terms[i], int(X.toarray()[0, i])) for i in np.where(X.toarray()[0] > 0)[0]])
print("D1 cell(s) with value 2:", [(terms[i], int(X.toarray()[0, i])) for i in np.where(X.toarray()[0] == 2)[0]])
for s in ["win free phone", "phone free win"]:
    print(f"{s!r}:", cv.transform([s]).toarray()[0].tolist())

heading("TASK 3 — TOKENS AND VOCABULARY SIZE")
token_counts = [len(cv.build_analyzer()(doc)) for doc in D]
print("Tokens per document:", token_counts)
print("Total token occurrences:", sum(token_counts))
print("Vocabulary size:", len(terms))
doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
print("Words in more than one document:", [(terms[i], int(doc_freq[i])) for i in range(len(terms)) if doc_freq[i] > 1])

heading("TASK 4 — PREPROCESSING SETTINGS ON D1")
for label, vec in [
    ("default", CountVectorizer()),
    ("lowercase=False", CountVectorizer(lowercase=False)),
    ('stop_words="english"', CountVectorizer(stop_words="english")),
]:
    vec.fit([D[0]])
    print(label, ":", vec.get_feature_names_out().tolist())
vec_stop = CountVectorizer(stop_words="english")
vec_stop.fit(D)
print('stop_words="english" on D1-D5:', vec_stop.get_feature_names_out().tolist())
print("Is 'not' in vocabulary?", "not" in vec_stop.vocabulary_)

heading("TASK 5 — LIMITING THE VOCABULARY")
for label, vec in [("min_df=2", CountVectorizer(min_df=2)), ("max_features=5", CountVectorizer(max_features=5))]:
    vec.fit(D)
    print(label, "vocabulary:", vec.get_feature_names_out().tolist())
    print(label, "size:", len(vec.vocabulary_))
print("Full vocabulary size:", len(terms))

heading("TASK 6 — N-GRAMS")
for ngram in [(1, 1), (1, 2)]:
    vec = CountVectorizer(ngram_range=ngram)
    vec.fit([D[4]])
    print(f"D5 ngram_range={ngram}:", vec.get_feature_names_out().tolist())
for ngram in [(1, 1), (1, 2), (1, 3)]:
    vec = CountVectorizer(ngram_range=ngram)
    vec.fit(D)
    print(f"D1-D5 ngram_range={ngram}: vocabulary size={len(vec.vocabulary_)}")

heading("TASK 7 — TF-IDF")
tfidf = TfidfVectorizer()
T = tfidf.fit_transform(D)
tfidf_terms = tfidf.get_feature_names_out()
print("Features:", tfidf_terms.tolist())
print("Shape:", T.shape)
print(np.round(T.toarray(), 2))
counts = X.toarray()
weights = T.toarray()
for doc_idx in [0, 1]:
    print(f"D{doc_idx+1} word | count | tf-idf")
    for j, word in enumerate(terms):
        if counts[doc_idx, j] > 0:
            tfidf_j = tfidf.vocabulary_[word]
            print(f"{word} | {counts[doc_idx, j]} | {weights[doc_idx, tfidf_j]:.6f}")

heading("TASK 8 — TEXTBOOK TF-IDF BY HAND FOR D2")
N_docs = len(D)
for word in ["cash", "today"]:
    j = cv.vocabulary_[word]
    count = int(X.toarray()[1, j])
    df = int(doc_freq[j])
    tf = count  # raw-count textbook TF
    idf = math.log(N_docs / df)
    product = tf * idf
    print(f"{word}: count={count}, df={df}, N={N_docs}, TF={tf}, IDF=ln({N_docs}/{df})={idf:.6f}, TF-IDF={product:.6f}")

heading("TASK 9 — SPARSITY")
arr = X.toarray()
total = arr.size
nonzero = int(np.count_nonzero(arr))
zero = total - nonzero
print("Total cells:", total)
print("Non-zero cells:", nonzero)
print("Zero cells:", zero)
print("Percentage zeros:", f"{100 * zero / total:.2f}%")

heading("TASK 10 — FIT ON TRAINING, TRANSFORM TEST")
train_vec = CountVectorizer()
train_X = train_vec.fit_transform(D[:4])
print("Training vocabulary:", train_vec.get_feature_names_out().tolist())
new_X = train_vec.transform(N)
for name, doc, row in zip(["N1", "N2"], N, new_X.toarray()):
    print(f"{name} {doc!r}: {row.tolist()}")
    tokens = train_vec.build_analyzer()(doc)
    oov = sorted(set(tokens) - set(train_vec.vocabulary_))
    print(f"{name} OOV:", oov)

heading("TASK 11 — DATA LEAKAGE DEMONSTRATION")
train_only = CountVectorizer().fit(D[:4])
train_plus_test = CountVectorizer().fit(D[:4] + N)
print("Train-only vocabulary:", train_only.get_feature_names_out().tolist())
print("Train + test vocabulary:", train_plus_test.get_feature_names_out().tolist())
leaked = sorted(set(train_plus_test.vocabulary_) - set(train_only.vocabulary_))
print("Words introduced by test data:", leaked)

heading("TASK 12 — BAG OF WORDS DOES NOT CAPTURE MEANING")
rvec = CountVectorizer()
RX = rvec.fit_transform(R)
rterms = rvec.get_feature_names_out()
show_matrix(RX, rterms, ["R1", "R2", "R3"])
print("Cosine similarity matrix:")
print(np.round(cosine_similarity(RX), 4))
print("R1-R2 similarity:", cosine_similarity(RX[0], RX[1])[0, 0])
print("R1-R3 similarity:", cosine_similarity(RX[0], RX[2])[0, 0])
