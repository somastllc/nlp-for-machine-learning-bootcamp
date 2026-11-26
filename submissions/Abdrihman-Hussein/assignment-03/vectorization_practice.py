# ============================================================
# Assignment 3 - Vectorization
# Student: Abdrihman Hussein
# Vectorizes a small corpus and prints the result of every step.
#
# Run from anywhere:
#   python vectorization_practice.py > output.txt
# ============================================================

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=250)

# --------------------------------
# The corpus from the assignment
# --------------------------------
DOCUMENTS = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]

NEW_TEXTS = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?",
]

REVIEWS = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible",
]


def print_table(matrix, row_labels, column_labels, digits=None):
    """Print a small labelled matrix without needing pandas."""
    if digits is None:
        cells = [[str(int(value)) for value in row] for row in matrix]
    else:
        cells = [[f"{value:.{digits}f}" for value in row] for row in matrix]

    widths = []
    for column_index, label in enumerate(column_labels):
        width = len(str(label))
        for row in cells:
            width = max(width, len(row[column_index]))
        widths.append(width)

    header = " " * 9 + " ".join(
        f"{label:>{width}}" for label, width in zip(column_labels, widths)
    )
    print(header)
    for row_label, row in zip(row_labels, cells):
        line = " ".join(f"{value:>{width}}" for value, width in zip(row, widths))
        print(f"{row_label:<9}{line}")


# --------------------------------
# 1) One-hot encoding of single words
# Each vocabulary word gets a vector with a single 1 in its own position.
# A sentence becomes a list with one vector per word, so its length depends
# on the number of words in the sentence.
# --------------------------------
ONE_HOT_VOCABULARY = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot(word, vocabulary):
    """Return a list of 0/1 values for one word."""
    return [1 if vocabulary_word == word else 0 for vocabulary_word in vocabulary]


print("\n=== 1) ONE-HOT ENCODING ===")
print("Vocabulary:", ONE_HOT_VOCABULARY)

print("\nWord by word for 'free cash':")
for word in "free cash".split():
    print(f"  {word:<6} -> {one_hot(word, ONE_HOT_VOCABULARY)}")

print("\nSentence 'win free cash':")
sentence_vectors = []
for word in "win free cash".split():
    vector = one_hot(word, ONE_HOT_VOCABULARY)
    sentence_vectors.append(vector)
    print(f"  {word:<6} -> {vector}")
print("Number of vectors:", len(sentence_vectors), "(one per word)")

# --------------------------------
# 2) Bag of Words and the document-term matrix
# fit learns the vocabulary; each word becomes one column, each document one row.
# Word order is lost: "win free phone" and "phone free win" get the same vector.
# --------------------------------
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(DOCUMENTS)
count_vocabulary = count_vectorizer.get_feature_names_out()

print("\n=== 2) BAG OF WORDS AND THE DOCUMENT-TERM MATRIX ===")
print("Vocabulary:", count_vocabulary.tolist())
print("Shape (documents, words):", count_matrix.shape)
print("\nDocument-term matrix (rows = documents, columns = vocabulary):")
print_table(
    count_matrix.toarray(),
    [f"D{number}" for number in range(1, 6)],
    count_vocabulary,
)

win_index = count_vocabulary.tolist().index("win")
print("\nD1 row:", count_matrix.toarray()[0].tolist())
print("D1 cell for 'win':", count_matrix[0, win_index], "because WIN appears twice in D1")

print("\nTransform two new phrases:")
phrase_matrix = count_vectorizer.transform(["win free phone", "phone free win"])
print_table(phrase_matrix.toarray(), ["win free phone", "phone free win"], count_vocabulary)
print("Same vector:", np.array_equal(phrase_matrix.toarray()[0], phrase_matrix.toarray()[1]))

# --------------------------------
# 3) Tokens vs vocabulary size
# Tokens count every word occurrence; vocabulary size counts unique words.
# The vectorizer drops one-letter words and punctuation, so we count with
# the same analyser that the vectorizer uses.
# --------------------------------
analyser = count_vectorizer.build_analyzer()
tokens_per_document = [len(analyser(document)) for document in DOCUMENTS]
document_frequency = (count_matrix > 0).sum(axis=0).A1

print("\n=== 3) TOKENS vs VOCABULARY SIZE ===")
print("Tokens per document (vectorizer analyser):", tokens_per_document)
print("Total tokens (vectorizer analyser):", sum(tokens_per_document))
print("Total tokens (plain whitespace split):", sum(len(document.split()) for document in DOCUMENTS))
print("Vocabulary size:", len(count_vocabulary))

shared_words = [
    word for word, frequency in zip(count_vocabulary, document_frequency) if frequency > 1
]
print("Words that appear in more than one document:", shared_words)

# --------------------------------
# 4) Preprocessing inside the vectorizer
# By default CountVectorizer lowercases and drops punctuation and one-letter
# words. stop_words="english" removes common English words.
# --------------------------------
d1 = [DOCUMENTS[0]]
default_vectorizer = CountVectorizer().fit(d1)
no_lowercase_vectorizer = CountVectorizer(lowercase=False).fit(d1)
stopword_vectorizer = CountVectorizer(stop_words="english").fit(d1)

print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER (D1 ONLY) ===")
print("D1:", d1[0])
print("Default:              ", default_vectorizer.get_feature_names_out().tolist())
print("lowercase=False:      ", no_lowercase_vectorizer.get_feature_names_out().tolist())
print("stop_words='english': ", stopword_vectorizer.get_feature_names_out().tolist())

print("\nWhat happened to 'a':", "present" if "a" in default_vectorizer.vocabulary_ else "removed")
print("What happened to '!':", "present" if "!" in default_vectorizer.vocabulary_ else "removed")
print("Default forms of win:", [word for word in default_vectorizer.vocabulary_ if word.lower() == "win"])
print("lowercase=False forms of win:",
      [word for word in no_lowercase_vectorizer.vocabulary_ if word.lower() == "win"])
print("Stopwords removed from D1:",
      sorted(set(default_vectorizer.vocabulary_) - set(stopword_vectorizer.vocabulary_)))

stopword_all = CountVectorizer(stop_words="english").fit(DOCUMENTS)
print("\nstop_words='english' on D1-D5 vocabulary:", stopword_all.get_feature_names_out().tolist())
print("'not' removed from the vocabulary:", "not" not in stopword_all.vocabulary_)
print("D5 with stopwords removed:",
      stopword_all.build_analyzer()(DOCUMENTS[4]))

# --------------------------------
# 5) Limiting the vocabulary
# min_df=2 keeps words that appear in at least 2 documents.
# max_features=5 keeps only the 5 most frequent words.
# --------------------------------
min_df_vectorizer = CountVectorizer(min_df=2).fit(DOCUMENTS)
max_features_vectorizer = CountVectorizer(max_features=5).fit(DOCUMENTS)

print("\n=== 5) LIMITING THE VOCABULARY ===")
print("Full vocabulary size:", len(count_vocabulary))
print("Full vocabulary:      ", count_vocabulary.tolist())
print("min_df=2 size:", len(min_df_vectorizer.get_feature_names_out()))
print("min_df=2:     ", min_df_vectorizer.get_feature_names_out().tolist())
print("max_features=5 size:", len(max_features_vectorizer.get_feature_names_out()))
print("max_features=5:     ", max_features_vectorizer.get_feature_names_out().tolist())

# --------------------------------
# 6) N-grams
# Unigrams are single words; bigrams are pairs of neighbouring words.
# Bigrams keep part of the word order, for example "not good".
# --------------------------------
d5 = [DOCUMENTS[4]]
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1)).fit(d5)
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(d5)

print("\n=== 6) N-GRAMS ===")
print("D5:", d5[0])
print("Unigrams (1,1):        ", unigram_vectorizer.get_feature_names_out().tolist())
print("Unigrams + bigrams (1,2):", bigram_vectorizer.get_feature_names_out().tolist())

print("\nVocabulary size on D1-D5:")
for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram_range).fit(DOCUMENTS)
    print(f"  ngram_range={ngram_range}: {len(vectorizer.get_feature_names_out())}")

# --------------------------------
# 7) TF-IDF
# Same vocabulary and shape as counts, but each cell is a weight.
# Words found in fewer documents get higher weights.
# TF-IDF values are weights, not probabilities.
# --------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(DOCUMENTS)
tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out()

print("\n=== 7) TF-IDF ===")
print("TF-IDF matrix (rounded to 2 decimals):")
print_table(
    tfidf_matrix.toarray(),
    [f"D{number}" for number in range(1, 6)],
    tfidf_vocabulary,
    digits=2,
)

print("\nD1 words with count and TF-IDF weight:")
d1_counts = count_matrix.toarray()[0]
d1_weights = tfidf_matrix.toarray()[0]
for word, count, weight in zip(count_vocabulary, d1_counts, d1_weights):
    if count > 0:
        print(f"  {word:<10} count: {count} | tf-idf: {weight:.3f}")

print("\nD2 words with count and TF-IDF weight:")
d2_counts = count_matrix.toarray()[1]
d2_weights = tfidf_matrix.toarray()[1]
for word, count, weight in zip(count_vocabulary, d2_counts, d2_weights):
    if count > 0:
        print(f"  {word:<10} count: {count} | tf-idf: {weight:.3f}")

d1_ranking = sorted(
    [(word, weight) for word, weight in zip(count_vocabulary, d1_weights) if weight > 0],
    key=lambda pair: pair[1],
    reverse=True,
)
print("\nD1 words ranked by TF-IDF:", [(word, round(weight, 3)) for word, weight in d1_ranking])

# --------------------------------
# 8) TF-IDF by hand (textbook formula)
# TF = count / words in document, IDF = log(N / df), TF-IDF = TF x IDF.
# scikit-learn adjusts the formula slightly, so its numbers differ.
# --------------------------------
N = len(DOCUMENTS)
d2_tokens = analyser(DOCUMENTS[1])

print("\n=== 8) TF-IDF BY HAND FOR D2 ===")
print("D2:", DOCUMENTS[1])
print("D2 tokens:", d2_tokens)
print("Number of tokens in D2:", len(d2_tokens))
print("N (number of documents):", N)

for word in ["cash", "today"]:
    word_index = count_vocabulary.tolist().index(word)
    df = int(document_frequency[word_index])
    tf = d2_counts[word_index] / len(d2_tokens)
    idf = np.log(N / df)
    textbook_weight = tf * idf
    sklearn_weight = tfidf_matrix[1, word_index]
    print(f"\n  {word}:")
    print(f"    count in D2 = {int(d2_counts[word_index])}")
    print(f"    df = {df}")
    print(f"    TF = {int(d2_counts[word_index])} / {len(d2_tokens)} = {tf:.4f}")
    print(f"    IDF = log({N} / {df}) = {idf:.4f}")
    print(f"    textbook TF-IDF = {tf:.4f} x {idf:.4f} = {textbook_weight:.4f}")
    print(f"    scikit-learn TF-IDF = {sklearn_weight:.4f}")

# --------------------------------
# 9) Sparse matrix
# Most cells are zero, so scikit-learn stores only the non-zero cells.
# --------------------------------
total_cells = count_matrix.shape[0] * count_matrix.shape[1]
non_zero_cells = count_matrix.nnz
zero_cells = total_cells - non_zero_cells

print("\n=== 9) SPARSE MATRIX ===")
print("Total cells:   ", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:    ", zero_cells)
print("Percentage of zeros:", round(zero_cells / total_cells * 100, 2), "%")

# --------------------------------
# 10) Fit on training text, transform new text
# fit_transform learns the vocabulary from D1-D4 only.
# transform reuses it: new (out-of-vocabulary) words have no column.
# --------------------------------
train_vectorizer = CountVectorizer().fit(DOCUMENTS[:4])
train_vocabulary = train_vectorizer.get_feature_names_out()

print("\n=== 10) FIT ON D1-D4, TRANSFORM N1 AND N2 ===")
print("Training vocabulary:", train_vocabulary.tolist())

for label, text in zip(["N1", "N2"], NEW_TEXTS):
    vector = train_vectorizer.transform([text])
    print(f"\n{label}: {text}")
    print_table(vector.toarray(), [label], train_vocabulary)
    tokens = train_vectorizer.build_analyzer()(text)
    out_of_vocabulary = [token for token in tokens if token not in train_vectorizer.vocabulary_]
    print("Out-of-vocabulary words:", out_of_vocabulary)

# --------------------------------
# 11) Data leakage
# Fitting on D1-D4 + N1 + N2 lets test words into the vocabulary before
# evaluation. That is leakage: the vectorizer has seen the test set.
# --------------------------------
train_only_vectorizer = CountVectorizer().fit(DOCUMENTS[:4])
leaky_vectorizer = CountVectorizer().fit(DOCUMENTS[:4] + NEW_TEXTS)

train_only_words = set(train_only_vectorizer.get_feature_names_out())
leaky_words = set(leaky_vectorizer.get_feature_names_out())
leaked_words = sorted(leaky_words - train_only_words)

print("\n=== 11) DATA LEAKAGE ===")
print("Fitted on D1-D4 only:", sorted(train_only_words))
print("Fitted on D1-D4 + N1 + N2:", sorted(leaky_words))
print("Words that leaked from the test set:", leaked_words)

# --------------------------------
# 12) The main limitation: no meaning
# "tasty" and "delicious" mean almost the same, but they are separate,
# unrelated columns. Counts cannot see that "terrible" is the opposite.
# --------------------------------
review_vectorizer = CountVectorizer().fit(REVIEWS)
review_matrix = review_vectorizer.transform(REVIEWS)
review_vocabulary = review_vectorizer.get_feature_names_out()
similarity_matrix = cosine_similarity(review_matrix)

print("\n=== 12) NO MEANING: COUNTS AND COSINE SIMILARITY ===")
print("Vocabulary:", review_vocabulary.tolist())
print("\nReview vectors:")
print_table(review_matrix.toarray(), ["R1", "R2", "R3"], review_vocabulary)
print("\nCosine similarity matrix:")
print_table(similarity_matrix, ["R1", "R2", "R3"], ["R1", "R2", "R3"], digits=2)
print("R1 vs R2 (tasty vs delicious):", round(similarity_matrix[0, 1], 2))
print("R1 vs R3 (tasty vs terrible): ", round(similarity_matrix[0, 2], 2))
