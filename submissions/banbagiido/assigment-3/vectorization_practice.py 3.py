# Assignment 3 - Vectorization (practice script)
# Run: python vectorization_practice.py > output.txt

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=250)  # one matrix row per line

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
D = [
    "Win a FREE phone now! Reply WIN to claim.",             # D1
    "Free entry: win cash prizes today. Call now.",          # D2
    "Are we still meeting for lunch today?",                 # D3
    "Please send me the lecture notes before class.",        # D4
    "The lecture was not good, but the notes were great.",   # D5
]
N_TEXTS = [
    "Claim your free prize now",                             # N1
    "Is the lunch meeting moved to tomorrow?",               # N2
]
R = [
    "the food was tasty",       # R1
    "the food was delicious",   # R2
    "the food was terrible",    # R3
]
doc_names = ["D1", "D2", "D3", "D4", "D5"]


def vocab_of(vec):
    return vec.get_feature_names_out().tolist()


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ==================================================================
# 1) ONE-HOT ENCODING (plain Python, no library)
# ==================================================================
heading("1) ONE-HOT ENCODING (own encoder)")
one_hot_vocab = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot(word, vocabulary):
    """Return a vector with a single 1 at the word's position."""
    return [1 if v == word else 0 for v in vocabulary]


print("Vocabulary:", one_hot_vocab)
print("\nVectors for the sentence 'free cash':")
for w in "free cash".split():
    print(f"  {w:6s} -> {one_hot(w, one_hot_vocab)}")

print("\nVectors for the sentence 'win free cash':")
for w in "win free cash".split():
    print(f"  {w:6s} -> {one_hot(w, one_hot_vocab)}")
print("Number of vectors for 'free cash':    ", len("free cash".split()))
print("Number of vectors for 'win free cash':", len("win free cash".split()))

# ==================================================================
# 2) BAG OF WORDS
# ==================================================================
heading("2) BAG OF WORDS (CountVectorizer on D1-D5)")
cv = CountVectorizer()
X_count = cv.fit_transform(D)
vocab = vocab_of(cv)
counts = X_count.toarray()

print("Vocabulary:", vocab)
print("Shape (rows, columns):", X_count.shape)
print("Document-term matrix:")
print(counts)

print("\nCells with value 2:")
for i, j in zip(*np.where(counts == 2)):
    print(f"  {doc_names[i]}, word '{vocab[j]}' = {counts[i, j]}")

print("\n'win free phone':", cv.transform(["win free phone"]).toarray())
print("'phone free win':", cv.transform(["phone free win"]).toarray())
print("Same vector?    :", np.array_equal(
    cv.transform(["win free phone"]).toarray(),
    cv.transform(["phone free win"]).toarray()))

# ==================================================================
# 3) TOKENS vs VOCABULARY SIZE
# ==================================================================
heading("3) TOKENS vs VOCABULARY SIZE")
analyzer = cv.build_analyzer()  # same tokenization the vectorizer uses
tokens_per_doc = [analyzer(d) for d in D]
for name, toks in zip(doc_names, tokens_per_doc):
    print(f"{name}: {len(toks)} tokens -> {toks}")
total_tokens = sum(len(t) for t in tokens_per_doc)
print("\nTotal tokens (vectorizer's tokenization):", total_tokens)
print("Vocabulary size:                         ", len(vocab))

doc_freq = (counts > 0).sum(axis=0)  # df of every word
multi = [(w, int(df)) for w, df in zip(vocab, doc_freq) if df > 1]
print("\nWords that appear in more than one document (word, df):")
for w, df in multi:
    in_docs = [doc_names[i] for i in range(len(D)) if counts[i, vocab.index(w)] > 0]
    print(f"  {w:8s} df={df}  in {in_docs}")

# ==================================================================
# 4) PREPROCESSING INSIDE THE VECTORIZER
# ==================================================================
heading("4) PREPROCESSING INSIDE THE VECTORIZER (fit on D1 only)")
d1 = [D[0]]
print("D1:", D[0])
v_default = vocab_of(CountVectorizer().fit(d1))
v_nolower = vocab_of(CountVectorizer(lowercase=False).fit(d1))
v_stop = vocab_of(CountVectorizer(stop_words="english").fit(d1))
print("Default:              ", v_default)
print("lowercase=False:      ", v_nolower)
print("stop_words='english': ", v_stop)
print("Removed by stopword list (default minus stop_words):",
      sorted(set(v_default) - set(v_stop)))

print("\nstop_words='english' fitted on D1-D5:")
v_stop_all = vocab_of(CountVectorizer(stop_words="english").fit(D))
print("Vocabulary:", v_stop_all)
print("Removed from the full vocabulary:", sorted(set(vocab) - set(v_stop_all)))
print("Was 'not' removed?", "not" not in v_stop_all)
print("Vocabulary of D5 with stop_words='english':",
      vocab_of(CountVectorizer(stop_words="english").fit([D[4]])))

# ==================================================================
# 5) LIMITING THE VOCABULARY
# ==================================================================
heading("5) LIMITING THE VOCABULARY (D1-D5)")
v_min = CountVectorizer(min_df=2).fit(D)
v_max = CountVectorizer(max_features=5).fit(D)
print("Full vocabulary size:", len(vocab))
print("min_df=2        ->", len(vocab_of(v_min)), vocab_of(v_min))
print("max_features=5  ->", len(vocab_of(v_max)), vocab_of(v_max))
total_count_per_word = counts.sum(axis=0)
print("\nTotal count of each word (for max_features):")
print({w: int(c) for w, c in zip(vocab, total_count_per_word)})

# ==================================================================
# 6) N-GRAMS
# ==================================================================
heading("6) N-GRAMS")
d5 = [D[4]]
print("D5:", D[4])
print("ngram_range=(1, 1):", vocab_of(CountVectorizer(ngram_range=(1, 1)).fit(d5)))
print("ngram_range=(1, 2):", vocab_of(CountVectorizer(ngram_range=(1, 2)).fit(d5)))
print("\nVocabulary size on D1-D5:")
for rng in [(1, 1), (1, 2), (1, 3)]:
    size = len(vocab_of(CountVectorizer(ngram_range=rng).fit(D)))
    print(f"  ngram_range={rng}: {size}")

# ==================================================================
# 7) TF-IDF
# ==================================================================
heading("7) TF-IDF (TfidfVectorizer on D1-D5)")
tv = TfidfVectorizer()
X_tfidf = tv.fit_transform(D)
tf_vocab = vocab_of(tv)
tfidf = X_tfidf.toarray()
print("Vocabulary:", tf_vocab)
print("TF-IDF matrix (rounded to 2 decimals):")
print(tfidf.round(2))

for idx in [0, 1]:
    print(f"\n{doc_names[idx]}: {D[idx]}")
    print(f"  {'word':8s} {'count':>5s} {'tf-idf':>8s}")
    for j, w in enumerate(tf_vocab):
        if counts[idx, j] > 0:
            print(f"  {w:8s} {counts[idx, j]:5d} {tfidf[idx, j]:8.3f}")

print("\nscikit-learn IDF values (smooth_idf=True): ")
for w, idf in zip(tf_vocab, tv.idf_):
    print(f"  {w:8s} {idf:.3f}")

# ==================================================================
# 8) TF-IDF BY HAND (textbook formula) for D2
# ==================================================================
heading("8) TF-IDF BY HAND FOR D2 (textbook formula)")
N = len(D)
d2_tokens = analyzer(D[1])
n_words_d2 = len(d2_tokens)
print("D2 tokens:", d2_tokens)
print("Words in D2 (vectorizer's way):", n_words_d2)
print("N (number of documents):", N)
hand = {}
for w in ["cash", "today"]:
    count = d2_tokens.count(w)
    df = int(doc_freq[vocab.index(w)])
    tf = count / n_words_d2
    idf = np.log(N / df)
    hand[w] = tf * idf
    print(f"\nword '{w}'")
    print(f"  count in D2 = {count}")
    print(f"  df          = {df}")
    print(f"  TF          = {count}/{n_words_d2} = {tf}")
    print(f"  IDF         = ln({N}/{df}) = {idf:.4f}")
    print(f"  TF-IDF      = {tf} x {idf:.4f} = {tf * idf:.4f}")

print("\nTextbook ranking:     ", sorted(hand, key=hand.get, reverse=True))
sk = {w: tfidf[1, tf_vocab.index(w)] for w in ["cash", "today"]}
print("scikit-learn weights: ", {w: round(float(v), 3) for w, v in sk.items()})
print("scikit-learn ranking: ", sorted(sk, key=sk.get, reverse=True))

# ==================================================================
# 9) SPARSE MATRIX
# ==================================================================
heading("9) SPARSE MATRIX (count matrix from task 2)")
total_cells = X_count.shape[0] * X_count.shape[1]
nonzero = X_count.nnz
zeros = total_cells - nonzero
print("Matrix type:      ", type(X_count).__name__)
print("Total cells:      ", total_cells)
print("Non-zero cells:   ", nonzero)
print("Zero cells:       ", zeros)
print(f"Percentage zeros:  {100 * zeros / total_cells:.1f}%")

# ==================================================================
# 10) FIT AND TRANSFORM
# ==================================================================
heading("10) FIT ON D1-D4, TRANSFORM N1 AND N2")
train = D[:4]
vec = CountVectorizer()
X_train = vec.fit_transform(train)
train_vocab = vocab_of(vec)
print("Vocabulary (from D1-D4):", train_vocab)
print("Vocabulary size:", len(train_vocab))
an = vec.build_analyzer()
for name, text in zip(["N1", "N2"], N_TEXTS):
    print(f"\n{name}: {text}")
    print("  tokens:", an(text))
    print("  vector:", vec.transform([text]).toarray())
    oov = [t for t in an(text) if t not in train_vocab]
    print("  out-of-vocabulary words:", oov)
print("\n'prize' in vocabulary? ", "prize" in train_vocab)
print("'prizes' in vocabulary?", "prizes" in train_vocab)

# ==================================================================
# 11) DATA LEAKAGE
# ==================================================================
heading("11) DATA LEAKAGE")
clean = CountVectorizer().fit(D[:4])
leaky = CountVectorizer().fit(D[:4] + N_TEXTS)
print("Fitted on D1-D4 only (correct):")
print(" ", vocab_of(clean), "->", len(vocab_of(clean)), "words")
print("Fitted on D1-D4 + N1 + N2 (leakage):")
print(" ", vocab_of(leaky), "->", len(vocab_of(leaky)), "words")
leaked = sorted(set(vocab_of(leaky)) - set(vocab_of(clean)))
print("Leaked words:", leaked)

# ==================================================================
# 12) NO MEANING
# ==================================================================
heading("12) NO MEANING (R1-R3)")
rv = CountVectorizer()
X_r = rv.fit_transform(R)
print("Vocabulary:", vocab_of(rv))
for name, text, row in zip(["R1", "R2", "R3"], R, X_r.toarray()):
    print(f"{name} '{text}':", row)
sim = cosine_similarity(X_r)
print("\nCosine similarity matrix (rows/cols = R1, R2, R3):")
print(sim.round(3))
print(f"\nR1 vs R2 (tasty vs delicious): {sim[0, 1]:.3f}")
print(f"R1 vs R3 (tasty vs terrible):  {sim[0, 2]:.3f}")
print("R1 more similar to R2 than to R3?", bool(sim[0, 1] > sim[0, 2]))
