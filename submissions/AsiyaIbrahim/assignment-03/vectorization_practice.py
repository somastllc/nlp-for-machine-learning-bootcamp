"""Assignment 3 - Vectorization practice (Part B)."""
import math
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

D = {
    "D1": "Win a FREE phone now! Reply WIN to claim.",
    "D2": "Free entry: win cash prizes today. Call now.",
    "D3": "Are we still meeting for lunch today?",
    "D4": "Please send me the lecture notes before class.",
    "D5": "The lecture was not good, but the notes were great.",
}
N = {
    "N1": "Claim your free prize now",
    "N2": "Is the lunch meeting moved to tomorrow?",
}
R = {
    "R1": "the food was tasty",
    "R2": "the food was delicious",
    "R3": "the food was terrible",
}
docs = list(D.values())
names = list(D.keys())


def heading(t):
    print("\n" + t)
    print("=" * len(t))


def vocab_list(cv):
    return list(cv.get_feature_names_out())


# --------------------- Task 1
heading("TASK 1: One-hot encoding (own encoder, plain Python)")
VOCAB = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot(word, vocab=VOCAB):
    return [1 if word == v else 0 for v in vocab]


print("Vocabulary:", VOCAB)
print("\n-- 'free cash' --")
for w in "free cash".split():
    print(f"{w:6s} -> {one_hot(w)}")
print("\n-- 'win free cash' --")
for w in "win free cash".split():
    print(f"{w:6s} -> {one_hot(w)}")
print("\nNumber of vectors for 'free cash':", len("free cash".split()))
print("Number of vectors for 'win free cash':", len("win free cash".split()))

# --------------------- Task 2
heading("TASK 2: Bag of Words (CountVectorizer on D1-D5)")
cv = CountVectorizer()
X = cv.fit_transform(docs)
print("Vocabulary (word -> column index):")
print(dict(sorted(cv.vocabulary_.items(), key=lambda kv: kv[1])))
print("\nShape (documents x vocabulary):", X.shape)
print("\nColumns:", vocab_list(cv))
print("\nDocument-term matrix:")
for n, row in zip(names, X.toarray()):
    print(n, row)
d1 = X.toarray()[0]
twos = [w for w, c in zip(vocab_list(cv), d1) if c == 2]
print("\nWords with count 2 in D1:", twos)
print("\nTransform 'win free phone' and 'phone free win':")
a = cv.transform(["win free phone"]).toarray()[0]
b = cv.transform(["phone free win"]).toarray()[0]
print("win free phone :", a)
print("phone free win :", b)
print("Same vector?", bool((a == b).all()))

# --------------------- Task 3
heading("TASK 3: Tokens vs vocabulary size")
analyzer = cv.build_analyzer()
total_tokens = sum(len(analyzer(d)) for d in docs)
for n, d in zip(names, docs):
    print(n, analyzer(d))
print("\nTotal tokens (word occurrences):", total_tokens)
print("Vocabulary size:", len(cv.vocabulary_))
df = (X.toarray() > 0).sum(axis=0)
multi = [(w, int(c)) for w, c in zip(vocab_list(cv), df) if c > 1]
print("Words appearing in more than one document (word, df):", multi)

# ---------- Task 4
heading("TASK 4: Preprocessing inside the vectorizer")
cv_def = CountVectorizer().fit([D["D1"]])
cv_nolower = CountVectorizer(lowercase=False).fit([D["D1"]])
cv_stop = CountVectorizer(stop_words="english").fit([D["D1"]])
print("Default            :", vocab_list(cv_def))
print("lowercase=False    :", vocab_list(cv_nolower))
print("stop_words=english :", vocab_list(cv_stop))
removed = sorted(set(vocab_list(cv_def)) - set(vocab_list(cv_stop)))
print("\nWords removed by the stopword list:", removed)
cv_stop_all = CountVectorizer(stop_words="english").fit(docs)
print("\nstop_words='english' fit on D1-D5:", vocab_list(cv_stop_all))
print("Is 'not' in the vocabulary?", "not" in cv_stop_all.vocabulary_)
print("D5 after stopword removal:", cv_stop_all.build_analyzer()(D["D5"]))

# --------------------- Task 5
heading("TASK 5: Limiting the vocabulary")
cv_full = CountVectorizer().fit(docs)
cv_min = CountVectorizer(min_df=2).fit(docs)
cv_max = CountVectorizer(max_features=5).fit(docs)
print("Full vocabulary size:", len(cv_full.vocabulary_))
print("\nmin_df=2 -> size", len(cv_min.vocabulary_), vocab_list(cv_min))
print("\nmax_features=5 -> size", len(cv_max.vocabulary_), vocab_list(cv_max))

# --------------------- Task 6
heading("TASK 6: N-grams")
cv_u = CountVectorizer(ngram_range=(1, 1)).fit([D["D5"]])
cv_b = CountVectorizer(ngram_range=(1, 2)).fit([D["D5"]])
print("D5 (1,1):", vocab_list(cv_u))
print("\nD5 (1,2):", vocab_list(cv_b))
print("\nBigrams containing 'not':", [g for g in vocab_list(cv_b) if "not" in g.split()])
print("\nVocabulary size on D1-D5 for different ngram_range:")
for rng in [(1, 1), (1, 2), (1, 3)]:
    c = CountVectorizer(ngram_range=rng).fit(docs)
    print(f"ngram_range={rng}: {len(c.vocabulary_)}")

# --------------Task 7
heading("TASK 7: TF-IDF (TfidfVectorizer on D1-D5)")
tv = TfidfVectorizer()
T = tv.fit_transform(docs)
words = vocab_list(tv)
print("Columns:", words)
Xc = X.toarray()
Tt = T.toarray()
print("\nTF-IDF matrix (word rows, document columns; rounded to 2 decimals):")
print(f"{'Word':<12}" + "".join(f"{n:>8}" for n in names))
for j, w in enumerate(words):
    print(f"{w:<12}" + "".join(f"{value:>8.2f}" for value in Tt[:, j]))
for idx, n in [(0, "D1"), (1, "D2")]:
    print(f"\n{n}: word | count | tf-idf")
    for j, w in enumerate(words):
        if Xc[idx, j] > 0:
            print(f"  {w:8s} count={Xc[idx, j]}  tfidf={Tt[idx, j]:.4f}")
print("\nIDF values used by scikit-learn (smooth):")
for w, i in zip(words, tv.idf_):
    print(f"  {w:8s} {i:.4f}")

# ---------------- Task 8
heading("TASK 8: TF-IDF by hand for D2 (cash, today)")
d2_tokens = analyzer(D["D2"])
n_docs = len(docs)
len_d2 = len(d2_tokens)
print("D2 tokens:", d2_tokens, "-> length", len_d2)
print("Number of documents N =", n_docs)
by_hand = {}
for w in ["cash", "today"]:
    j = words.index(w)
    count = int(Xc[1, j])
    dfw = int((Xc[:, j] > 0).sum())
    tf = count / len_d2
    idf = math.log(n_docs / dfw)
    by_hand[w] = tf * idf
    print(f"\nWord: {w}")
    print(f"  count in D2 = {count}")
    print(f"  df          = {dfw}")
    print(f"  TF  = {count}/{len_d2} = {tf:.4f}")
    print(f"  IDF = ln({n_docs}/{dfw}) = {idf:.4f}")
    print(f"  TF-IDF = {tf:.4f} * {idf:.4f} = {tf * idf:.4f}")
print("\nTextbook ranking:", sorted(by_hand, key=by_hand.get, reverse=True))
sk = {w: Tt[1, words.index(w)] for w in ["cash", "today"]}
print("scikit-learn values:", {k: round(float(v), 4) for k, v in sk.items()})
print("scikit-learn ranking:", sorted(sk, key=sk.get, reverse=True))
print("Same ranking?", sorted(by_hand, key=by_hand.get, reverse=True)
      == sorted(sk, key=sk.get, reverse=True))

# ----------------- Task 9
heading("TASK 9: Sparse matrix statistics (count matrix from task 2)")
total = X.shape[0] * X.shape[1]
nonzero = X.nnz
zeros = total - nonzero
print("Total cells :", total)
print("Non-zero    :", nonzero)
print("Zero cells  :", zeros)
print(f"Percentage of zeros: {100 * zeros / total:.1f}%")

# ------------------ Task 10
heading("TASK 10: Fit and transform (fit on D1-D4, transform N1, N2)")
train = [D[k] for k in ["D1", "D2", "D3", "D4"]]
cv_tr = CountVectorizer().fit(train)
cols = vocab_list(cv_tr)
print("Vocabulary (fit on D1-D4):", cols)
print("Size:", len(cols))
for k, t in N.items():
    vec = cv_tr.transform([t]).toarray()[0]
    toks = cv_tr.build_analyzer()(t)
    oov = [w for w in toks if w not in cv_tr.vocabulary_]
    print(f"\n{k}: {t}")
    print("  tokens:", toks)
    print("  vector:", vec)
    print("  out-of-vocabulary words:", oov)
print("\n'prize' in vocabulary?", "prize" in cv_tr.vocabulary_)
print("'prizes' in vocabulary?", "prizes" in cv_tr.vocabulary_)

# ---------------------Task 11
heading("TASK 11: Data leakage")
correct = CountVectorizer().fit(train)
leaky = CountVectorizer().fit(train + list(N.values()))
print("Fit on D1-D4 only       :", vocab_list(correct), "-> size", len(correct.vocabulary_))
print("\nFit on D1-D4 + N1 + N2  :", vocab_list(leaky), "-> size", len(leaky.vocabulary_))
leaked = sorted(set(vocab_list(leaky)) - set(vocab_list(correct)))
print("\nWords leaked from the test set:", leaked)

# ---------------------Task 12
heading("TASK 12: No meaning (R1-R3)")
rev = list(R.values())
cv_r = CountVectorizer().fit(rev)
Xr = cv_r.transform(rev)
print("Vocabulary:", vocab_list(cv_r))
for k, row in zip(R.keys(), Xr.toarray()):
    print(k, row)
sim = cosine_similarity(Xr)
print("\nCosine similarity matrix (rows/cols = R1, R2, R3):")
print(np.round(sim, 3))
print(f"\nsim(R1,R2) tasty vs delicious = {sim[0, 1]:.3f}")
print(f"sim(R1,R3) tasty vs terrible  = {sim[0, 2]:.3f}")
print("Is R1 more similar to R2 than to R3?", bool(sim[0, 1] > sim[0, 2]))
