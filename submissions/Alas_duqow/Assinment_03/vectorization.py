# Assignment 3 - Vectorization (Part B)
# Run: python vectorization_practice.py > output.txt

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=200)

docs = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]
new_texts = ["Claim your free prize now", "Is the lunch meeting moved to tomorrow?"]
reviews = ["the food was tasty", "the food was delicious", "the food was terrible"]


def heading(n, title):
    print("\n" + "=" * 50)
    print(f"TASK {n}: {title}")
    print("=" * 50)


# ---- Task 1: one-hot ----
heading(1, "ONE-HOT ENCODING")
one_hot_vocab = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot(word):
    vec = [0] * len(one_hot_vocab)
    if word in one_hot_vocab:
        vec[one_hot_vocab.index(word)] = 1
    return vec


print("Vocabulary:", one_hot_vocab)
for w in "free cash".split():
    print(f"{w} -> {one_hot(w)}")
sentence = "win free cash".split()
print("Sentence:", sentence)
print("Vectors:", [one_hot(w) for w in sentence])
print("Number of vectors:", len(sentence))

# ---- Task 2: Bag of Words ----
heading(2, "BAG OF WORDS")
cv = CountVectorizer()
X = cv.fit_transform(docs)
terms = cv.get_feature_names_out().tolist()
counts = X.toarray()
print("Vocabulary:", terms)
print("Shape (rows, columns):", X.shape)
print("Document-term matrix:")
print(counts)
for j in np.where(counts[0] == 2)[0]:
    print(f"D1 cell with value 2: word '{terms[j]}'")
for t in ["win free phone", "phone free win"]:
    print(f"'{t}' ->", cv.transform([t]).toarray()[0].tolist())
print("Same vector?", np.array_equal(cv.transform(["win free phone"]).toarray(),
                                      cv.transform(["phone free win"]).toarray()))

# ---- Task 3: tokens vs vocabulary ----
heading(3, "TOKENS VS VOCABULARY SIZE")
analyzer = cv.build_analyzer()
token_lists = [analyzer(d) for d in docs]
print("Tokens per document:", [len(t) for t in token_lists])
print("Total tokens:", sum(len(t) for t in token_lists))
print("Vocabulary size:", len(terms))
doc_freq = {w: int(np.count_nonzero(counts[:, k])) for k, w in enumerate(terms)}
print("Words in more than one document:", [w for w, d in doc_freq.items() if d > 1])

# ---- Task 4: preprocessing ----
heading(4, "PREPROCESSING INSIDE THE VECTORIZER")
settings = [("Default", {}), ("lowercase=False", {"lowercase": False}),
            ("stop_words='english'", {"stop_words": "english"})]
d1_default = set(CountVectorizer().fit([docs[0]]).get_feature_names_out())
for name, kw in settings:
    print(f"{name}:", CountVectorizer(**kw).fit([docs[0]]).get_feature_names_out().tolist())
d1_stop = set(CountVectorizer(stop_words="english").fit([docs[0]]).get_feature_names_out())
print("Removed by stop list from D1:", sorted(d1_default - d1_stop))
sv = CountVectorizer(stop_words="english").fit(docs)
print("stop_words='english' on D1-D5:", sv.get_feature_names_out().tolist())
print("'not' still in vocabulary?", "not" in sv.vocabulary_)

# ---- Task 5: limiting vocabulary ----
heading(5, "LIMITING THE VOCABULARY")
print("Full vocabulary size:", len(terms))
for name, kw in [("min_df=2", {"min_df": 2}), ("max_features=5", {"max_features": 5})]:
    names = CountVectorizer(**kw).fit(docs).get_feature_names_out().tolist()
    print(f"{name}: {names} (size {len(names)})")

# ---- Task 6: n-grams ----
heading(6, "N-GRAMS")
for r in [(1, 1), (1, 2)]:
    print(f"D5 ngram_range={r}:", CountVectorizer(ngram_range=r).fit([docs[4]]).get_feature_names_out().tolist())
for r in [(1, 1), (1, 2), (1, 3)]:
    print(f"D1-D5 ngram_range={r} vocabulary size:",
          len(CountVectorizer(ngram_range=r).fit(docs).get_feature_names_out()))

# ---- Task 7: TF-IDF ----
heading(7, "TF-IDF")
tv = TfidfVectorizer()
T = tv.fit_transform(docs)
tterms = tv.get_feature_names_out().tolist()
tfidf = T.toarray()
print("Vocabulary:", tterms)
print("TF-IDF matrix (rounded to 2 decimals):")
print(tfidf.round(2))
for i, name in [(0, "D1"), (1, "D2")]:
    print(f"\n{name}: word | count | TF-IDF")
    for j, w in enumerate(terms):
        if counts[i, j] > 0:
            print(f"  {w:8s} | {int(counts[i, j])} | {tfidf[i, tterms.index(w)]:.3f}")

# ---- Task 8: TF-IDF by hand ----
heading(8, "TF-IDF BY HAND FOR D2")
N = len(docs)
total_d2 = int(counts[1].sum())
print("N =", N, "| words in D2 (as the vectorizer counts them) =", total_d2)
for w in ["cash", "today"]:
    j = terms.index(w)
    c = int(counts[1, j])
    df = int(np.count_nonzero(counts[:, j]))
    tf = c / total_d2
    idf = np.log(N / df)
    print(f"{w}: count={c}, df={df}, TF={c}/{total_d2}={tf:.4f}, "
          f"IDF=ln({N}/{df})={idf:.4f}, TF-IDF={tf * idf:.4f}")
    print(f"   scikit-learn TF-IDF = {tfidf[1, tterms.index(w)]:.4f}")

# ---- Task 9: sparse matrix ----
heading(9, "SPARSE MATRIX")
total = counts.size
nz = int(np.count_nonzero(counts))
print("Total cells:", total)
print("Non-zero cells:", nz)
print("Zero cells:", total - nz)
print(f"Percentage of zeros: {(total - nz) / total * 100:.2f}%")

# ---- Task 10: fit and transform ----
heading(10, "FIT ON TRAIN, TRANSFORM NEW TEXT")
tr = CountVectorizer().fit(docs[:4])
known = set(tr.get_feature_names_out())
print("Training vocabulary:", sorted(known))
an = tr.build_analyzer()
for name, text in [("N1", new_texts[0]), ("N2", new_texts[1])]:
    print(f"\n{name}: {text}")
    print("Vector:", tr.transform([text]).toarray()[0].tolist())
    print("Out-of-vocabulary words:", [t for t in an(text) if t not in known])

# ---- Task 11: data leakage ----
heading(11, "DATA LEAKAGE")
v_train = set(CountVectorizer().fit(docs[:4]).get_feature_names_out())
v_all = set(CountVectorizer().fit(docs[:4] + new_texts).get_feature_names_out())
print("Fitted on D1-D4 only:", sorted(v_train))
print("Fitted on D1-D4 + N1 + N2:", sorted(v_all))
print("Leaked words:", sorted(v_all - v_train))

# ---- Task 12: no meaning ----
heading(12, "NO MEANING")
rv = CountVectorizer()
R = rv.fit_transform(reviews)
print("Vocabulary:", rv.get_feature_names_out().tolist())
for k, row in enumerate(R.toarray(), 1):
    print(f"R{k}:", row.tolist())
sim = cosine_similarity(R)
print("Cosine similarity matrix:")
print(sim.round(4))
print("R1-R2 (tasty vs delicious):", round(sim[0, 1], 4))
print("R1-R3 (tasty vs terrible): ", round(sim[0, 2], 4))
