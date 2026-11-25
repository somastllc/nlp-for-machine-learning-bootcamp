# Assignment 3 — Vectorization (practical script)

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=240, precision=2, suppress=True)

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

# --------------------------------
# 1) One-hot encoding
# Plain Python: a 1 marks the word's vocabulary position.
# A sentence has one vector per word, so longer sentences have more vectors.
# --------------------------------
one_hot_vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]
print("\n=== 1) ONE-HOT ENCODING ===")
print("Vocabulary:", one_hot_vocabulary)
for sentence in ["free cash", "win free cash"]:
    print("Sentence:", sentence)
    for word in sentence.split():
        vector = [1 if word == item else 0 for item in one_hot_vocabulary]
        print(" ", word, "->", vector)
    print("Number of word vectors:", len(sentence.split()))

# --------------------------------
# 2) Bag of Words (document-term matrix)
# fit_transform learns words and counts each word in every document.
# The word "win" appears twice in D1.
# --------------------------------
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)
vocabulary = count_vectorizer.get_feature_names_out()
counts_array = count_matrix.toarray()
print("\n=== 2) BAG OF WORDS ===")
for i, document in enumerate(documents, start=1):
    print(f"D{i}:", document)
print("Vocabulary:", vocabulary.tolist())
print("Shape (documents, features):", count_matrix.shape)
print("Document-term matrix (rows D1-D5, columns vocabulary order):")
print(counts_array)
print("D1 'win' count:", counts_array[0, count_vectorizer.vocabulary_["win"]])
print("'win free phone':", count_vectorizer.transform(["win free phone"]).toarray())
print("'phone free win':", count_vectorizer.transform(["phone free win"]).toarray())

# --------------------------------
# 3) Tokens vs vocabulary size
# Count occurrences after the same tokenizer used by CountVectorizer.
# Document frequency (df) counts in how many documents a term appears.
# --------------------------------
analyzer = count_vectorizer.build_analyzer()
all_tokens = [word for doc in documents for word in analyzer(doc)]
document_frequency = np.asarray((count_matrix > 0).sum(axis=0)).ravel()
repeated_across_docs = [(word, int(df)) for word, df in zip(vocabulary, document_frequency) if df > 1]
print("\n=== 3) TOKENS vs VOCABULARY SIZE ===")
print("Total tokens (vectorizer's tokenizer):", len(all_tokens))
print("Vocabulary size:", len(vocabulary))
print("Words in more than one document (word, df):", repeated_across_docs)

# --------------------------------
# 4) Preprocessing inside CountVectorizer
# It lowercases, drops punctuation and one-character tokens by default.
# The built-in English stopwords list can also remove useful words like "not".
# --------------------------------
print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")
print("D1:", documents[0])
for label, opts in [
    ("Default", {}),
    ("lowercase=False", {"lowercase": False}),
    ("stop_words='english'", {"stop_words": "english"}),
]:
    features = CountVectorizer(**opts).fit(documents[:1]).get_feature_names_out().tolist()
    print(label + ":", features)
full_default = set(vocabulary)
full_stop = set(CountVectorizer(stop_words="english").fit(documents).get_feature_names_out())
stop_features_d1 = set(CountVectorizer(stop_words="english").fit(documents[:1]).get_feature_names_out())
print("Default D1 words removed by English stopwords:", sorted(set(analyzer(documents[0])) - stop_features_d1))
print("Full-corpus vocabulary with English stopwords:", sorted(full_stop))
print("'not' in full default vocabulary?", "not" in full_default)
print("'not' in full stopword-filtered vocabulary?", "not" in full_stop)

# --------------------------------
# 5) Limiting vocabulary
# min_df=2 keeps words seen in at least 2 different documents.
# max_features=5 keeps the 5 highest-frequency corpus terms.
# --------------------------------
print("\n=== 5) LIMITING THE VOCABULARY ===")
for label, params in [
    ("Full vocabulary", {}),
    ("min_df=2", {"min_df": 2}),
    ("max_features=5", {"max_features": 5}),
]:
    words = CountVectorizer(**params).fit(documents).get_feature_names_out().tolist()
    print(f"{label}: size={len(words)}, words={words}")

# --------------------------------
# 6) N-grams
# Bigrams keep pairs such as "not good" if the tokenizer places them together.
# Larger n-grams quickly increase the number of vocabulary columns.
# --------------------------------
print("\n=== 6) N-GRAMS ===")
print("D5:", documents[4])
for upper in [1, 2]:
    features = CountVectorizer(ngram_range=(1, upper)).fit([documents[4]]).get_feature_names_out().tolist()
    print(f"D5 ngram_range=(1, {upper}):", features)
for upper in [1, 2, 3]:
    features = CountVectorizer(ngram_range=(1, upper)).fit(documents).get_feature_names_out()
    print(f"All D1-D5 ngram_range=(1, {upper}) vocabulary size:", len(features))

# --------------------------------
# 7) TF-IDF
# TF-IDF uses corpus-wide document frequency and L2 normalization.
# All rows and columns match the count matrix with the same settings.
# --------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_words = tfidf_vectorizer.get_feature_names_out()
tfidf_values = tfidf_matrix.toarray()
assert vocabulary.tolist() == tfidf_words.tolist()
print("\n=== 7) TF-IDF ===")
print("Vocabulary:", tfidf_words.tolist())
print("TF-IDF matrix (rounded to 2 decimals):")
print(np.round(tfidf_values, 2))
for i in [0, 1]:
    print(f"D{i + 1} count vs TF-IDF (word -> count, weight):")
    for word, count, weight in zip(vocabulary, counts_array[i], tfidf_values[i]):
        if count:
            print(f"  {word:12s} count={count} tf-idf={weight:.4f}")

# --------------------------------
# 8) TF-IDF by hand (textbook formula)
# TF = count / document's tokens; IDF = ln(N / df).
# The scikit-learn formula includes smoothing, +1 and L2 normalization.
# --------------------------------
print("\n=== 8) TF-IDF BY HAND (D2) ===")
N = len(documents)
d2_token_count = len(analyzer(documents[1]))
print("N (documents):", N)
print("D2 token count (CountVectorizer tokenizer):", d2_token_count)
for word in ["cash", "today"]:
    idx = count_vectorizer.vocabulary_[word]
    count = int(counts_array[1, idx])
    df = int(document_frequency[idx])
    tf = count / d2_token_count
    idf = np.log(N / df)
    tfidf = tf * idf
    print(f"{word}: count={count}, df={df}, TF={tf:.6f}, IDF=ln({N}/{df})={idf:.6f}, TF-IDF={tfidf:.6f}, sklearn={tfidf_values[1, idx]:.6f}")

# --------------------------------
# 9) Sparse matrix
# Store only nonzero cells for efficient memory use.
# --------------------------------
total_cells = count_matrix.shape[0] * count_matrix.shape[1]
nonzero_cells = count_matrix.nnz
zero_cells = total_cells - nonzero_cells
print("\n=== 9) SPARSE MATRIX ===")
print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print("Zero percentage:", f"{zero_cells / total_cells * 100:.2f}%")
print("Storage format:", count_matrix.getformat())

# --------------------------------
# 10) Fit on D1-D4 only, transform N1 and N2
# New words have no vocabulary column; similar word forms are not merged.
# --------------------------------
train_documents = documents[:4]
train_vectorizer = CountVectorizer()
X_train = train_vectorizer.fit_transform(train_documents)
train_vocabulary = set(train_vectorizer.get_feature_names_out())
train_analyzer = train_vectorizer.build_analyzer()
print("\n=== 10) FIT ON TRAIN, TRANSFORM TEST ===")
print("Training vocabulary:", train_vectorizer.get_feature_names_out().tolist())
print("Training matrix shape:", X_train.shape)
for i, new_text in enumerate(new_texts, start=1):
    print(f"N{i}:", new_text)
    print(" Vector:", train_vectorizer.transform([new_text]).toarray())
    print(" OOV words:", sorted(set(train_analyzer(new_text)) - train_vocabulary))
print("'prize' and 'prizes' are different word forms; stemming/lemmatization can reduce this mismatch.")

# --------------------------------
# 11) Data leakage
# Fitting a vocabulary on test texts exposes test-only words before evaluation.
# --------------------------------
leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(train_documents + new_texts)
leaky_vocab = set(leaky_vectorizer.get_feature_names_out())
print("\n=== 11) DATA LEAKAGE ===")
print("Training-only vocabulary:", sorted(train_vocabulary))
print("Training + test vocabulary:", sorted(leaky_vocab))
print("Words leaked from N1/N2:", sorted(leaky_vocab - train_vocabulary))

# --------------------------------
# 12) Main limitation: no meaning
# Count vectors count shared identical words, not the meaning of synonyms.
# Compare review vectors using cosine similarity.
# --------------------------------
meaning_vectorizer = CountVectorizer()
review_matrix = meaning_vectorizer.fit_transform(reviews)
print("\n=== 12) NO MEANING: COSINE SIMILARITY ===")
print("Reviews:", reviews)
print("Vocabulary:", meaning_vectorizer.get_feature_names_out().tolist())
print("Count vectors (R1-R3):")
print(review_matrix.toarray())
print("Cosine similarity matrix (R1-R3):")
print(np.round(cosine_similarity(review_matrix), 4))
