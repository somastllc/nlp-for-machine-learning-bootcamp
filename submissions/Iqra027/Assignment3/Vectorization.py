import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=200, suppress=True)

# Training corpus (D1-D5)
D1 = "Win a FREE phone now! Reply WIN to claim."
D2 = "Free entry: win cash prizes today. Call now."
D3 = "Are we still meeting for lunch today?"
D4 = "Please send me the lecture notes before class."
D5 = "The lecture was not good, but the notes were great."
docs = [D1, D2, D3, D4, D5]

# New texts (N1-N2)
N1 = "Claim your free prize now"
N2 = "Is the lunch meeting moved to tomorrow?"

# Reviews (R1-R3)
R1 = "the food was tasty"
R2 = "the food was delicious"
R3 = "the food was terrible"
reviews = [R1, R2, R3]


def one_hot_encode(word, vocab):
    return [1 if term == word else 0 for term in vocab]


# --------------------------------
# 1) One-hot encoding
# --------------------------------
print("=== 1) ONE-HOT ENCODING ===")
vocab_1 = ["cash", "class", "free", "lunch", "notes", "win"]
print("Vocabulary:", vocab_1)

for w in ["free", "cash"]:
    print(f"Word '{w}':", one_hot_encode(w, vocab_1))

phrase_1 = "win free cash"
print(f"One-hot vectors for '{phrase_1}' (length {len(phrase_1.split())}):")
for w in phrase_1.split():
    print(f"  {w} -> {one_hot_encode(w, vocab_1)}")


# --------------------------------
# 2) Bag of Words
# --------------------------------
print("\n=== 2) BAG OF WORDS ===")
cv_all = CountVectorizer()
X_counts = cv_all.fit_transform(docs)
vocab_all = cv_all.get_feature_names_out().tolist()

print("Vocabulary:", vocab_all)
print("Shape (rows, cols):", X_counts.shape)
print("Document-term matrix:\n", X_counts.toarray())
print("D1 count for 'win':", X_counts.toarray()[0, vocab_all.index("win")])

for q in ["win free phone", "phone free win"]:
    print(f"Transform '{q}':", cv_all.transform([q]).toarray())


# --------------------------------
# 3) Tokens vs vocabulary size
# --------------------------------
print("\n=== 3) TOKENS VS VOCABULARY SIZE ===")
analyzer = cv_all.build_analyzer()
tokens_d1_d5 = [tok for doc in docs for tok in analyzer(doc)]
doc_freq = (X_counts.toarray() > 0).sum(axis=0)
multi_doc_words = [vocab_all[i] for i in range(len(vocab_all)) if doc_freq[i] > 1]

print("Total tokens in D1-D5:", len(tokens_d1_d5))
print("Vocabulary size:      ", len(vocab_all))
print("Words in >1 document: ", multi_doc_words)


# --------------------------------
# 4) Preprocessing inside the vectorizer
# --------------------------------
print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")
print("D1 default:             ", CountVectorizer().fit([D1]).get_feature_names_out().tolist())
print("D1 lowercase=False:     ", CountVectorizer(lowercase=False).fit([D1]).get_feature_names_out().tolist())
print("D1 stop_words='english':", CountVectorizer(stop_words="english").fit([D1]).get_feature_names_out().tolist())

cv_sw = CountVectorizer(stop_words="english").fit(docs)
print("D1-D5 stop_words='english':", cv_sw.get_feature_names_out().tolist())
print("Is 'not' in D1-D5 stop_words='english' vocab?", "not" in cv_sw.get_feature_names_out())


# --------------------------------
# 5) Limiting the vocabulary
# --------------------------------
print("\n=== 5) LIMITING THE VOCABULARY ===")
cv_min_df = CountVectorizer(min_df=2).fit(docs)
cv_max_feat = CountVectorizer(max_features=5).fit(docs)

print("Full vocab size:", len(vocab_all))
print("min_df=2 vocab: ", cv_min_df.get_feature_names_out().tolist(), "| size:", len(cv_min_df.get_feature_names_out()))
print("max_feat=5 voc:", cv_max_feat.get_feature_names_out().tolist(), "| size:", len(cv_max_feat.get_feature_names_out()))


# --------------------------------
# 6) N-grams
# --------------------------------
print("\n=== 6) N-GRAMS ===")
print("D5 ngram_range=(1, 1):", CountVectorizer(ngram_range=(1, 1)).fit([D5]).get_feature_names_out().tolist())
print("D5 ngram_range=(1, 2):", CountVectorizer(ngram_range=(1, 2)).fit([D5]).get_feature_names_out().tolist())

for ngram_r in [(1, 1), (1, 2), (1, 3)]:
    v_size = len(CountVectorizer(ngram_range=ngram_r).fit(docs).get_feature_names_out())
    print(f"D1-D5 ngram_range={ngram_r} vocab size:", v_size)


# --------------------------------
# 7) TF-IDF
# --------------------------------
print("\n=== 7) TF-IDF ===")
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(docs)
vocab_tfidf = tfidf_vec.get_feature_names_out().tolist()

print("TF-IDF matrix (rounded to 2 decimals):\n", np.round(X_tfidf.toarray(), 2))

for doc_idx, label in [(0, "D1"), (1, "D2")]:
    print(f"\n{label} word counts vs TF-IDF weights:")
    row_counts = X_counts.toarray()[doc_idx]
    row_weights = X_tfidf.toarray()[doc_idx]
    for i, term in enumerate(vocab_tfidf):
        if row_counts[i] > 0:
            print(
                f"  {term:10} | count: {row_counts[i]} | df: {doc_freq[i]} "
                f"| tfidf: {row_weights[i]:.4f} (round2: {round(float(row_weights[i]), 2)})"
            )


# --------------------------------
# 8) TF-IDF by hand (D2: 'cash' and 'today')
# --------------------------------
print("\n=== 8) TF-IDF BY HAND FOR D2 ===")
d2_tokens = analyzer(D2)
n_docs = len(docs)
d2_len = len(d2_tokens)
print("D2 tokens:", d2_tokens, "| count of tokens:", d2_len, "| total docs N:", n_docs)

for target_word in ["cash", "today"]:
    tf_val = d2_tokens.count(target_word) / d2_len
    df_val = int(doc_freq[vocab_all.index(target_word)])
    idf_val = np.log(n_docs / df_val)
    tfidf_hand = tf_val * idf_val
    print(
        f"  {target_word:6} -> tf={tf_val:.4f}, df={df_val}, "
        f"idf={idf_val:.4f}, tf*idf={tfidf_hand:.4f}"
    )


# --------------------------------
# 9) Sparse matrix
# --------------------------------
print("\n=== 9) SPARSE MATRIX ===")
total_cells = X_counts.shape[0] * X_counts.shape[1]
non_zero_cells = int(X_counts.nnz)
zero_cells = total_cells - non_zero_cells
pct_zeros = (zero_cells / total_cells) * 100

print("Total cells:    ", total_cells)
print("Non-zero cells: ", non_zero_cells)
print("Zero cells:     ", zero_cells)
print(f"Percentage zero:{pct_zeros:.2f}%")


# --------------------------------
# 10) Fit and transform (train D1-D4, test N1-N2)
# --------------------------------
print("\n=== 10) FIT AND TRANSFORM ===")
cv_train = CountVectorizer()
cv_train.fit(docs[:4])
train_vocab = cv_train.get_feature_names_out().tolist()
train_vocab_set = set(train_vocab)

print("Train vocabulary (D1-D4):", train_vocab)
for name, text in [("N1", N1), ("N2", N2)]:
    vec = cv_train.transform([text]).toarray()[0]
    oov = [tok for tok in analyzer(text) if tok not in train_vocab_set]
    print(f"{name}: '{text}'")
    print("  Vector:", vec.tolist())
    print("  OOV words:", oov)


# --------------------------------
# 11) Data leakage
# --------------------------------
print("\n=== 11) DATA LEAKAGE ===")
cv_leaky = CountVectorizer().fit(docs[:4] + [N1, N2])
leaky_vocab = cv_leaky.get_feature_names_out().tolist()
leaked_words = sorted(set(leaky_vocab) - train_vocab_set)

print("Vocab fitted on D1-D4 only:       ", train_vocab)
print("Vocab fitted on D1-D4 + N1 + N2:   ", leaky_vocab)
print("Leaked words from test set:      ", leaked_words)


# --------------------------------
# 12) No meaning (R1-R3)
# --------------------------------
print("\n=== 12) NO MEANING ===")
cv_rev = CountVectorizer()
X_rev = cv_rev.fit_transform(reviews)
sim_matrix = cosine_similarity(X_rev)

print("Reviews vocabulary:", cv_rev.get_feature_names_out().tolist())
print("Reviews count matrix:\n", X_rev.toarray())
print("Cosine similarity matrix (R1, R2, R3):\n", np.round(sim_matrix, 4))