import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=220, suppress=True)

documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]

new_texts = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?",
]

reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible",
]

# ---------------------------------------------------------------------------
# 1) One-hot encoding
# ---------------------------------------------------------------------------
oh_vocab = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot_encode(word, vocab):
    return [1 if term == word else 0 for term in vocab]


print("\n=== 1) ONE-HOT ENCODING ===")
print("Vocabulary:", oh_vocab)
for w in ["free", "cash"]:
    print(f"Word '{w}' -> {one_hot_encode(w, oh_vocab)}")

phrase_1 = ["win", "free", "cash"]
print(f"Vectors for 'win free cash' ({len(phrase_1)} words -> {len(phrase_1)} vectors):")
for w in phrase_1:
    print(f"  {w}: {one_hot_encode(w, oh_vocab)}")

# ---------------------------------------------------------------------------
# 2) Bag of Words
# ---------------------------------------------------------------------------
cv = CountVectorizer()
dtm = cv.fit_transform(documents)
vocab = cv.get_feature_names_out().tolist()

print("\n=== 2) BAG OF WORDS ===")
print("Vocabulary:", vocab)
print("Shape (documents, features):", dtm.shape)
print("Document-Term Matrix (D1-D5):\n", dtm.toarray())

v_win_free_phone = cv.transform(["win free phone"]).toarray()[0]
v_phone_free_win = cv.transform(["phone free win"]).toarray()[0]
print("Vector 'win free phone':", v_win_free_phone)
print("Vector 'phone free win':", v_phone_free_win)
print("Are they identical?", np.array_equal(v_win_free_phone, v_phone_free_win))

# ---------------------------------------------------------------------------
# 3) Tokens vs vocabulary size
# ---------------------------------------------------------------------------
analyzer = cv.build_analyzer()
all_doc_tokens = []
for doc in documents:
    all_doc_tokens.extend(analyzer(doc))

doc_presence = (dtm.toarray() > 0).sum(axis=0)
multi_doc_words = [vocab[i] for i, df_val in enumerate(doc_presence) if df_val > 1]

print("\n=== 3) TOKENS VS VOCABULARY SIZE ===")
print("Total tokens (D1-D5):", len(all_doc_tokens))
print("Vocabulary size:", len(vocab))
print("Words appearing in >1 document:", multi_doc_words)

# ---------------------------------------------------------------------------
# 4) Preprocessing inside the vectorizer
# ---------------------------------------------------------------------------
d1_only = [documents[0]]
cv_default = CountVectorizer().fit(d1_only)
cv_nocase = CountVectorizer(lowercase=False).fit(d1_only)
cv_stop_d1 = CountVectorizer(stop_words="english").fit(d1_only)

print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")
print("D1 default vocab:", cv_default.get_feature_names_out().tolist())
print("D1 lowercase=False vocab:", cv_nocase.get_feature_names_out().tolist())
print("D1 stop_words='english' vocab:", cv_stop_d1.get_feature_names_out().tolist())

removed_by_stop = sorted(
    set(cv_default.get_feature_names_out()) - set(cv_stop_d1.get_feature_names_out())
)
print("Removed by stop_words='english' on D1:", removed_by_stop)

cv_stop_all = CountVectorizer(stop_words="english").fit(documents)
vocab_stop_all = cv_stop_all.get_feature_names_out().tolist()
print("Is 'not' in D1-D5 vocab with stop_words='english'?", "not" in vocab_stop_all)

# ---------------------------------------------------------------------------
# 5) Limiting the vocabulary
# ---------------------------------------------------------------------------
cv_mindf2 = CountVectorizer(min_df=2).fit(documents)
cv_maxf5 = CountVectorizer(max_features=5).fit(documents)

print("\n=== 5) LIMITING THE VOCABULARY ===")
print("Full vocabulary size:", len(vocab))
print(
    "min_df=2 vocab:",
    cv_mindf2.get_feature_names_out().tolist(),
    "| size:",
    len(cv_mindf2.get_feature_names_out()),
)
print(
    "max_features=5 vocab:",
    cv_maxf5.get_feature_names_out().tolist(),
    "| size:",
    len(cv_maxf5.get_feature_names_out()),
)

# ---------------------------------------------------------------------------
# 6) N-grams
# ---------------------------------------------------------------------------
d5_only = [documents[4]]
cv_d5_1_1 = CountVectorizer(ngram_range=(1, 1)).fit(d5_only)
cv_d5_1_2 = CountVectorizer(ngram_range=(1, 2)).fit(d5_only)

print("\n=== 6) N-GRAMS ===")
print("D5 ngram_range=(1, 1):", cv_d5_1_1.get_feature_names_out().tolist())
print("D5 ngram_range=(1, 2):", cv_d5_1_2.get_feature_names_out().tolist())

for rng in [(1, 1), (1, 2), (1, 3)]:
    v_ng = CountVectorizer(ngram_range=rng).fit(documents)
    print(f"D1-D5 ngram_range={rng} -> vocab size: {len(v_ng.get_feature_names_out())}")

# ---------------------------------------------------------------------------
# 7) TF-IDF
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer()
tfidf_mat = tfidf.fit_transform(documents)
tfidf_vocab = tfidf.get_feature_names_out().tolist()
counts_arr = dtm.toarray()
weights_arr = tfidf_mat.toarray()

print("\n=== 7) TF-IDF ===")
print("TF-IDF matrix (rounded to 2 decimals):\n", np.round(weights_arr, 2))

for idx, label in [(0, "D1"), (1, "D2")]:
    print(f"\n{label} non-zero terms (Count vs TF-IDF):")
    for j, term in enumerate(tfidf_vocab):
        if counts_arr[idx, j] > 0:
            print(f"  {term:10s} | count={counts_arr[idx, j]} | tfidf={weights_arr[idx, j]:.4f}")

# ---------------------------------------------------------------------------
# 8) TF-IDF by hand (D2: 'cash' and 'today')
# ---------------------------------------------------------------------------
N = len(documents)
d2_tokens = analyzer(documents[1])
words_in_d2 = len(d2_tokens)

print("\n=== 8) TF-IDF BY HAND (D2) ===")
print(f"N = {N}, words in D2 = {words_in_d2}")
for term in ["cash", "today"]:
    j = vocab.index(term)
    tf_d2 = counts_arr[1, j] / words_in_d2
    df_t = int((counts_arr[:, j] > 0).sum())
    idf_hand = np.log(N / df_t)
    tfidf_hand = tf_d2 * idf_hand
    print(
        f"  {term:8s} | count={counts_arr[1, j]} | df={df_t} | "
        f"TF={tf_d2:.4f} | IDF={idf_hand:.4f} | TF-IDF(hand)={tfidf_hand:.4f} | "
        f"sklearn={weights_arr[1, j]:.4f}"
    )

# ---------------------------------------------------------------------------
# 9) Sparse matrix
# ---------------------------------------------------------------------------
total_cells = dtm.shape[0] * dtm.shape[1]
nnz_cells = dtm.nnz
zero_cells = total_cells - nnz_cells
zero_pct = (zero_cells / total_cells) * 100.0

print("\n=== 9) SPARSE MATRIX ===")
print("Total cells:", total_cells)
print("Non-zero cells:", nnz_cells)
print("Zero cells:", zero_cells)
print(f"Percentage of zeros: {zero_pct:.2f}%")

# ---------------------------------------------------------------------------
# 10) Fit and transform (Train: D1-D4, Test: N1-N2)
# ---------------------------------------------------------------------------
train_d1_d4 = documents[:4]
cv_train = CountVectorizer().fit(train_d1_d4)
train_vocab_set = set(cv_train.get_feature_names_out())

print("\n=== 10) FIT AND TRANSFORM (D1-D4 -> N1, N2) ===")
print("Train vocab size (D1-D4):", len(train_vocab_set))
for name, txt in zip(["N1", "N2"], new_texts):
    vec = cv_train.transform([txt]).toarray()[0]
    oov_words = [tok for tok in analyzer(txt) if tok not in train_vocab_set]
    print(f"{name} ('{txt}') vector:\n  {vec}")
    print(f"{name} OOV words: {oov_words}")

# ---------------------------------------------------------------------------
# 11) Data leakage
# ---------------------------------------------------------------------------
cv_leaky = CountVectorizer().fit(train_d1_d4 + new_texts)
leaky_vocab = cv_leaky.get_feature_names_out().tolist()
leaked_words = sorted(set(leaky_vocab) - train_vocab_set)

print("\n=== 11) DATA LEAKAGE ===")
print(f"Vocab fitted on D1-D4 only (size {len(train_vocab_set)}): {sorted(train_vocab_set)}")
print(f"Vocab fitted on D1-D4 + N1 + N2 (size {len(leaky_vocab)}): {leaky_vocab}")
print("Leaked words from test set:", leaked_words)

# ---------------------------------------------------------------------------
# 12) No meaning (Reviews R1-R3)
# ---------------------------------------------------------------------------
cv_rev = CountVectorizer()
rev_mat = cv_rev.fit_transform(reviews).toarray()
rev_vocab = cv_rev.get_feature_names_out().tolist()
sim_mat = cosine_similarity(rev_mat)

print("\n=== 12) NO MEANING (REVIEWS R1-R3) ===")
print("Reviews vocab:", rev_vocab)
for idx, r in enumerate(reviews):
    print(f"  R{idx + 1} ('{r}'): {rev_mat[idx]}")
print("Cosine similarity matrix (R1, R2, R3):\n", np.round(sim_mat, 4))
print(f"sim(R1, R2) = {sim_mat[0, 1]:.4f} | sim(R1, R3) = {sim_mat[0, 2]:.4f}")