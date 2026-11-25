import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Documents (training corpus)
D1 = "Win a FREE phone now! Reply WIN to claim."
D2 = "Free entry: win cash prizes today. Call now."
D3 = "Are we still meeting for lunch today?"
D4 = "Please send me the lecture notes before class."
D5 = "The lecture was not good, but the notes were great."

train_docs = [D1, D2, D3, D4, D5]
doc_names = ["D1", "D2", "D3", "D4", "D5"]

# New texts (test set)
N1 = "Claim your free prize now"
N2 = "Is the lunch meeting moved to tomorrow?"

# Reviews
R1 = "the food was tasty"
R2 = "the food was delicious"
R3 = "the food was terrible"


# Step 1: One-hot encoding
print("=== Step 1: One-hot encoding ===")
vocab_oh = ["cash", "class", "free", "lunch", "notes", "win"]
word_to_idx = {w: i for i, w in enumerate(vocab_oh)}

def one_hot_word(word):
    vec = [0] * len(vocab_oh)
    if word in word_to_idx:
        vec[word_to_idx[word]] = 1
    return vec

for w in ["free", "cash"]:
    print(f"{w}: {one_hot_word(w)}")

phrase_wfc = ["win", "free", "cash"]
print("win free cash ->", [one_hot_word(w) for w in phrase_wfc])


# Step 2: Bag of Words (CountVectorizer on D1-D5)
print("\n=== Step 2: Bag of Words ===")
vec_bow = CountVectorizer()
X_bow = vec_bow.fit_transform(train_docs)
vocab_bow = vec_bow.get_feature_names_out()
dtm_bow = X_bow.toarray()

print("Vocabulary:", list(vocab_bow))
print("Shape:", dtm_bow.shape)
print("Document-Term Matrix:\n", dtm_bow)

for name, text in [("win free phone", "win free phone"), ("phone free win", "phone free win")]:
    print(f"Transform '{name}':", vec_bow.transform([text]).toarray()[0])


# Step 3: Tokens vs vocabulary size
print("\n=== Step 3: Tokens vs vocabulary size ===")
analyzer_default = vec_bow.build_analyzer()
doc_tokens = [analyzer_default(d) for d in train_docs]
total_tokens = sum(len(toks) for toks in doc_tokens)
vocab_size = len(vocab_bow)

print("Total tokens (word occurrences) in D1-D5:", total_tokens)
print("Vocabulary size:", vocab_size)

multi_doc_words = [
    term for j, term in enumerate(vocab_bow)
    if np.count_nonzero(dtm_bow[:, j]) > 1
]
print("Words appearing in >1 document:", multi_doc_words)


# Step 4: Preprocessing inside the vectorizer
print("\n=== Step 4: Preprocessing inside the vectorizer ===")
for label, kw in [
    ("default", {}),
    ("lowercase=False", {"lowercase": False}),
    ("stop_words='english'", {"stop_words": "english"}),
]:
    v = CountVectorizer(**kw).fit([D1])
    print(f"D1 vocab ({label}):", list(v.get_feature_names_out()))

v_stop_all = CountVectorizer(stop_words="english").fit(train_docs)
stop_vocab_all = set(v_stop_all.get_feature_names_out())
print("'not' in D1-D5 vocabulary with stop_words='english'?", "not" in stop_vocab_all)


# Step 5: Limiting the vocabulary
print("\n=== Step 5: Limiting the vocabulary ===")
v_mindf2 = CountVectorizer(min_df=2).fit(train_docs)
v_maxfeat5 = CountVectorizer(max_features=5).fit(train_docs)

print(f"Full vocab size: {len(vocab_bow)}")
print(f"min_df=2 vocab (size {len(v_mindf2.get_feature_names_out())}):", list(v_mindf2.get_feature_names_out()))
print(f"max_features=5 vocab (size {len(v_maxfeat5.get_feature_names_out())}):", list(v_maxfeat5.get_feature_names_out()))


# Step 6: N-grams
print("\n=== Step 6: N-grams ===")
for rng in [(1, 1), (1, 2)]:
    v_d5 = CountVectorizer(ngram_range=rng).fit([D5])
    print(f"D5 vocab ngram_range={rng}:", list(v_d5.get_feature_names_out()))

for rng in [(1, 1), (1, 2), (1, 3)]:
    v_all = CountVectorizer(ngram_range=rng).fit(train_docs)
    print(f"D1-D5 vocab size for ngram_range={rng}: {len(v_all.get_feature_names_out())}")


# Step 7: TF-IDF
print("\n=== Step 7: TF-IDF ===")
vec_tfidf = TfidfVectorizer()
X_tfidf = vec_tfidf.fit_transform(train_docs).toarray()
print("TF-IDF matrix (rounded to 2 decimals):\n", np.round(X_tfidf, 2))

for idx, name in [(0, "D1"), (1, "D2")]:
    print(f"\n{name} terms (count, tfidf):")
    for j, term in enumerate(vocab_bow):
        if dtm_bow[idx, j] > 0:
            print(f"  {term}: count={dtm_bow[idx, j]}, tfidf={X_tfidf[idx, j]:.4f} (round={round(X_tfidf[idx, j], 2)})")


# Step 8: TF-IDF by hand (D2: cash and today)
print("\n=== Step 8: TF-IDF by hand ===")
N_docs = len(train_docs)
for term in ["cash", "today"]:
    j = list(vocab_bow).index(term)
    tf_d2 = dtm_bow[1, j]
    df_t = int(np.count_nonzero(dtm_bow[:, j]))
    idf_hand = np.log(N_docs / df_t)
    tfidf_hand = tf_d2 * idf_hand
    print(f"{term}: tf(D2)={tf_d2}, df={df_t}, N={N_docs}, IDF=ln({N_docs}/{df_t})={idf_hand:.4f}, TF-IDF={tfidf_hand:.4f}, sklearn_round2={round(X_tfidf[1, j], 2)}")


# Step 9: Sparse matrix
print("\n=== Step 9: Sparse matrix ===")
total_cells = dtm_bow.size
nonzero_cells = int(np.count_nonzero(dtm_bow))
zero_cells = total_cells - nonzero_cells
pct_zeros = (zero_cells / total_cells) * 100
print(f"Total cells: {total_cells}, Non-zero: {nonzero_cells}, Zeros: {zero_cells}, Zeros %: {pct_zeros:.2f}%")


# Step 10: Fit and transform (D1-D4 train, N1-N2 test)
print("\n=== Step 10: Fit and transform ===")
vec_14 = CountVectorizer().fit(train_docs[:4])
vocab_14 = set(vec_14.get_feature_names_out())
for name, text in [("N1", N1), ("N2", N2)]:
    vec_out = vec_14.transform([text]).toarray()[0]
    toks = analyzer_default(text)
    oov = [t for t in toks if t not in vocab_14]
    print(f"{name} vector:", vec_out)
    print(f"{name} OOV words:", oov)


# Step 11: Data leakage
print("\n=== Step 11: Data leakage ===")
vocab_train_only = set(CountVectorizer().fit(train_docs[:4]).get_feature_names_out())
vocab_with_test = set(CountVectorizer().fit(train_docs[:4] + [N1, N2]).get_feature_names_out())
leaked = sorted(vocab_with_test - vocab_train_only)
print("Vocab D1-D4 only (sorted):", sorted(vocab_train_only))
print("Vocab D1-D4 + N1 + N2 (sorted):", sorted(vocab_with_test))
print("Leaked words from test set:", leaked)


# Step 12: No meaning (R1-R3 cosine similarity)
print("\n=== Step 12: No meaning ===")
vec_rev = CountVectorizer()
X_rev = vec_rev.fit_transform([R1, R2, R3]).toarray()
print("Reviews vocabulary:", list(vec_rev.get_feature_names_out()))
print("R1, R2, R3 vectors:\n", X_rev)
sim_matrix = cosine_similarity(X_rev)
print("Cosine similarity matrix:\n", np.round(sim_matrix, 4))