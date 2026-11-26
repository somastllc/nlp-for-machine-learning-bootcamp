import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=200)

D = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]
N = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?",
]
R = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible",
]


def tokens(text):
    """Lowercase word tokens without punctuation (matches CountVectorizer's default pattern)."""
    return re.findall(r"\b\w\w+\b", text.lower())


# ---------------------------------------------------------------
print("\n=== 1) One-hot encoding ===")
one_hot_vocab = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot(word):
    return [1 if w == word else 0 for w in one_hot_vocab]


print("Vocabulary:", one_hot_vocab)
for word in "free cash".split():
    print(f"{word} -> {one_hot(word)}")
print()
for word in "win free cash".split():
    print(f"{word} -> {one_hot(word)}")

# Each word gets one vector, so 3 words give 3 vectors. Each vector has one
# slot per vocabulary word, so its length is 6 whatever the sentence length.

# ---------------------------------------------------------------
print("\n=== 2) Bag of Words ===")
count_vec = CountVectorizer()
count_matrix = count_vec.fit_transform(D)
vocab = count_vec.get_feature_names_out().tolist()
print("Vocabulary:", vocab)
print("Shape (documents, terms):", count_matrix.shape)
print(count_matrix.toarray())
print("'win free phone':", count_vec.transform(["win free phone"]).toarray())
print("'phone free win':", count_vec.transform(["phone free win"]).toarray())
print("Same vector:", np.array_equal(
    count_vec.transform(["win free phone"]).toarray(),
    count_vec.transform(["phone free win"]).toarray()))


# ---------------------------------------------------------------
print("\n=== 3) Tokens vs vocabulary size ===")
all_tokens = [w for d in D for w in d.split()]
print("Total tokens (whitespace split):", len(all_tokens))
print("Vocabulary size:", len(vocab))
doc_sets = [set(tokens(d)) for d in D]
repeated = [w for w in vocab if sum(w in s for s in doc_sets) > 1]
print("Words in more than one document:", repeated)


# ---------------------------------------------------------------
print("\n=== 4) Preprocessing inside the vectorizer (D1 only) ===")
d1 = [D[0]]
print("Default:           ", CountVectorizer().fit(d1).get_feature_names_out().tolist())
print("lowercase=False:   ", CountVectorizer(lowercase=False).fit(d1).get_feature_names_out().tolist())
print("stop_words=english:", CountVectorizer(stop_words="english").fit(d1).get_feature_names_out().tolist())
stop_all = CountVectorizer(stop_words="english").fit(D).get_feature_names_out().tolist()
print("stop_words=english on D1-D5 keeps 'not':", "not" in stop_all)
print("D5 words kept after stopword filter:", [w for w in tokens(D[4]) if w in stop_all])


# ---------------------------------------------------------------
print("\n=== 5) Limiting the vocabulary ===")
min_df_vec = CountVectorizer(min_df=2).fit(D)
max_f_vec = CountVectorizer(max_features=5).fit(D)
print("Full vocabulary size:", len(vocab))
print("min_df=2:      ", min_df_vec.get_feature_names_out().tolist(),
    "| size:", len(min_df_vec.get_feature_names_out()))
print("max_features=5:", max_f_vec.get_feature_names_out().tolist(),
    "| size:", len(max_f_vec.get_feature_names_out()))


# ---------------------------------------------------------------
print("\n=== 6) N-grams ===")
d5 = [D[4]]
uni = CountVectorizer(ngram_range=(1, 1)).fit(d5).get_feature_names_out().tolist()
bi = CountVectorizer(ngram_range=(1, 2)).fit(d5).get_feature_names_out().tolist()
print("D5 unigrams:            ", uni)
print("D5 unigrams and bigrams:", bi)
for r in [(1, 1), (1, 2), (1, 3)]:
    size = len(CountVectorizer(ngram_range=r).fit(D).get_feature_names_out())
    print(f"ngram_range={r} vocabulary size on D1-D5: {size}")

# ---------------------------------------------------------------
print("\n=== 7) TF-IDF ===")
tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(D).toarray()
tf_vocab = tfidf_vec.get_feature_names_out().tolist()
print(np.round(tfidf, 2))
for i in [0, 1]:
    print(f"\nD{i + 1}: {D[i]}")
    counts = count_vec.transform([D[i]]).toarray()[0]
    for j, w in enumerate(vocab):
        if counts[j] > 0:
            print(f"  {w:10} count={counts[j]}  tf-idf={tfidf[i][tf_vocab.index(w)]:.2f}")
# ---------------------------------------------------------------
print("\n=== 8) TF-IDF by hand for D2 ===")
N_docs = len(D)
d2_tokens = count_vec.build_analyzer()(D[1])
df = (count_matrix.toarray() > 0).sum(axis=0)
print("D2 tokens:", d2_tokens, "| words:", len(d2_tokens), "| N:", N_docs)
for w in ["cash", "today"]:
    tf_count = d2_tokens.count(w)
    tf = tf_count / len(d2_tokens)
    dfw = int(df[vocab.index(w)])
    idf = np.log(N_docs / dfw)
    print(f"{w}: count={tf_count} TF={tf:.4f} df={dfw} IDF={idf:.4f} TF-IDF={tf * idf:.4f}")


# ---------------------------------------------------------------
print("\n=== 9) Sparse matrix ===")
total = count_matrix.shape[0] * count_matrix.shape[1]
nonzero = count_matrix.nnz
zeros = total - nonzero
print("Total cells:", total)
print("Non-zero cells:", nonzero)
print("Zero cells:", zeros)
print(f"Percentage of zeros: {zeros / total * 100:.1f}%")

# ---------------------------------------------------------------
print("\n=== 10) Fit on D1-D4, transform N1 and N2 ===")
train_vec = CountVectorizer().fit(D[:4])
train_vocab = set(train_vec.get_feature_names_out())
train_an = train_vec.build_analyzer()
for name, text in zip(["N1", "N2"], N):
    vec = train_vec.transform([text]).toarray()
    oov = sorted({t for t in train_an(text) if t not in train_vocab})
    print(f"{name}: {text}")
    print("  vector:", vec)
    print("  out-of-vocabulary:", oov)

# ---------------------------------------------------------------
print("\n=== 11) Data leakage ===")
v_train = CountVectorizer().fit(D[:4])
v_all = CountVectorizer().fit(D[:4] + N)
a = set(v_train.get_feature_names_out())
b = set(v_all.get_feature_names_out())
print("Fitted on D1-D4:      ", sorted(a), "| size:", len(a))
print("Fitted on D1-D4 + N:  ", sorted(b), "| size:", len(b))
print("Leaked from test set:", sorted(b - a))

# ---------------------------------------------------------------
print("\n=== 12) No meaning: reviews ===")
rev_vec = CountVectorizer()
rev = rev_vec.fit_transform(R).toarray()
print("Vocabulary:", rev_vec.get_feature_names_out().tolist())
for label, row in zip(["R1", "R2", "R3"], rev):
    print(label, row)
sim = cosine_similarity(rev)
print("Cosine similarity:")
print(np.round(sim, 2))
print("R1 vs R2:", round(sim[0, 1], 2), "| R1 vs R3:", round(sim[0, 2], 2))

