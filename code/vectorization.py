# Lesson 3 — Vectorization (teaching script)
# Follows lessons/03-vectorization.md step by step.
# Run: python code/vectorization.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

np.set_printoptions(linewidth=200)  # print each matrix row on one line

documents = [
    "win free cash now",
    "free prize claim now",
    "see you in class today",
    "please send the class notes",
]

# --------------------------------
# 1) One-hot encoding of single words
# Each vocabulary word gets a vector with a single 1 in its own position.
# --------------------------------
one_hot_vocabulary = ["class", "free", "notes", "prize"]

print("\n=== 1) ONE-HOT ENCODING ===")
print("Vocabulary:", one_hot_vocabulary)
for word in "free prize".split():
    one_hot = [1 if vocabulary_word == word else 0 for vocabulary_word in one_hot_vocabulary]
    print(word, "->", one_hot)

# --------------------------------
# 2) Bag of Words and the document-term matrix
# fit learns the vocabulary; each word becomes one column, each document one row.
# Word order is lost: "free prize" and "prize free" get the same vector.
# --------------------------------
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print("\n=== 2) BAG OF WORDS ===")
print("Documents: ", documents)
print("Vocabulary:", count_vectorizer.get_feature_names_out().tolist())
print("Shape (rows, columns):", count_matrix.shape)
print(count_matrix.toarray())
print("'free prize':", count_vectorizer.transform(["free prize"]).toarray())
print("'prize free':", count_vectorizer.transform(["prize free"]).toarray())

# --------------------------------
# 3) Tokens vs vocabulary size
# Tokens count every word occurrence; vocabulary size counts unique words (columns).
# --------------------------------
all_tokens = " ".join(documents).split()

print("\n=== 3) TOKENS vs VOCABULARY SIZE ===")
print("Total tokens:   ", len(all_tokens))
print("Vocabulary size:", len(count_vectorizer.get_feature_names_out()))

# --------------------------------
# 4) The vectorizer does its own preprocessing
# By default CountVectorizer lowercases, drops punctuation and one-letter words.
# stop_words="english" removes common words — check what it removes.
# --------------------------------
message = ["FREE Prize! Call the office NOW"]

print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")
print("Message:              ", message[0])
print("Default:              ", CountVectorizer().fit(message).get_feature_names_out().tolist())
print("lowercase=False:      ", CountVectorizer(lowercase=False).fit(message).get_feature_names_out().tolist())
print("stop_words='english': ", CountVectorizer(stop_words="english").fit(message).get_feature_names_out().tolist())

# --------------------------------
# 5) Limiting the vocabulary: min_df and max_features
# min_df=2 keeps words found in at least 2 documents.
# max_features=5 keeps only the 5 most frequent words (ties keep vocabulary order).
# --------------------------------
print("\n=== 5) LIMITING THE VOCABULARY ===")
print("Full vocabulary size:", len(count_vectorizer.get_feature_names_out()))
print("min_df=2:            ", CountVectorizer(min_df=2).fit(documents).get_feature_names_out().tolist())
print("max_features=5:      ", CountVectorizer(max_features=5).fit(documents).get_feature_names_out().tolist())

# --------------------------------
# 6) N-grams
# ngram_range=(1, 2) adds bigrams (two neighboring words) to the single words.
# The default settings drop the one-letter word "a", so "not good" becomes a bigram.
# --------------------------------
sentence = ["not a good movie"]

print("\n=== 6) N-GRAMS ===")
print("Sentence:            ", sentence[0])
print("Unigrams:            ", CountVectorizer(ngram_range=(1, 1)).fit(sentence).get_feature_names_out().tolist())
print("Unigrams and bigrams:", CountVectorizer(ngram_range=(1, 2)).fit(sentence).get_feature_names_out().tolist())

# --------------------------------
# 7) TF-IDF
# Same vocabulary and shape as counts, but each cell is a weight.
# Words found in fewer documents get higher weights. Weights are not probabilities.
# --------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

vocabulary = count_vectorizer.get_feature_names_out()
counts = count_matrix.toarray()[0]
weights = tfidf_matrix.toarray()[0]

print("\n=== 7) TF-IDF ===")
print(tfidf_matrix.toarray().round(2))
print("Count vs TF-IDF for:", documents[0])
for word, count, weight in zip(vocabulary, counts, weights):
    if count > 0:
        print(" ", word, "-> count:", count, "| tf-idf:", round(weight, 3))

# --------------------------------
# 8) TF-IDF by hand (textbook formula)
# TF = count / words in document, IDF = log(N / df), TF-IDF = TF x IDF.
# scikit-learn adjusts this formula slightly, so its numbers differ,
# but the rarer word still gets the higher weight.
# --------------------------------
N = len(documents)
words_in_document = len(documents[0].split())

print("\n=== 8) TF-IDF BY HAND FOR:", documents[0], "===")
for word, df in [("win", 1), ("free", 2)]:
    tf = 1 / words_in_document
    idf = np.log(N / df)
    print(word, "| df =", df, "| TF =", tf, "| IDF =", round(idf, 3), "| TF-IDF =", round(tf * idf, 3))

# --------------------------------
# 9) Sparse matrices
# Most cells are zero, so scikit-learn stores only the non-zero cells.
# --------------------------------
total_cells = count_matrix.shape[0] * count_matrix.shape[1]

print("\n=== 9) SPARSE MATRIX ===")
print("Total cells:   ", total_cells)
print("Non-zero cells:", count_matrix.nnz)
print("Zero cells:    ", total_cells - count_matrix.nnz)

# --------------------------------
# 10) Fit on training text, transform new text
# fit_transform learns the vocabulary from training text only.
# transform reuses it: new (out-of-vocabulary) words have no column and are ignored.
# --------------------------------
train_texts = ["free prize", "class notes"]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)

print("\n=== 10) FIT ON TRAIN, TRANSFORM NEW TEXT ===")
print("Vocabulary (from training text):", vectorizer.get_feature_names_out().tolist())
print("Train matrix:")
print(X_train.toarray())
print("'free notes':      ", vectorizer.transform(["free notes"]).toarray())
print("'free meeting':    ", vectorizer.transform(["free meeting"]).toarray())
print("'meeting tomorrow':", vectorizer.transform(["meeting tomorrow"]).toarray())

# --------------------------------
# 11) Data leakage
# Suppose the test set contains "free meeting". Fitting on ALL text lets the
# test word "meeting" into the vocabulary before evaluation.
# --------------------------------
leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(train_texts + ["free meeting"])

print("\n=== 11) DATA LEAKAGE ===")
print("Fitted on training text only:", vectorizer.get_feature_names_out().tolist())
print("Fitted on train + test text: ", leaky_vectorizer.get_feature_names_out().tolist())

# --------------------------------
# 12) The main limitation: no meaning
# "good" and "great" mean almost the same, but they are separate, unrelated columns.
# Word embeddings such as Word2Vec (Lesson 5) address this.
# --------------------------------
reviews = ["the movie was good", "the movie was great"]
meaning_vectorizer = CountVectorizer()
meaning_matrix = meaning_vectorizer.fit_transform(reviews)

print("\n=== 12) NO MEANING: 'good' vs 'great' ===")
print("Vocabulary:", meaning_vectorizer.get_feature_names_out().tolist())
print("'the movie was good': ", meaning_matrix.toarray()[0])
print("'the movie was great':", meaning_matrix.toarray()[1])