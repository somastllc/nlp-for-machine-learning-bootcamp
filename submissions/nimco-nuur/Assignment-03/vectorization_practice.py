# Assignment 3 – Vectorization
# Part B – Practical

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# DOCUMENTS
documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

new_texts = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]

reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]


def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# TASK 1: ONE-HOT ENCODING
# ============================================================

heading("TASK 1: ONE-HOT ENCODING")

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot(word):
    return [int(w == word) for w in vocabulary]


print("Vocabulary:", vocabulary)

print("\nVectors for 'free cash':")

for word in ["free", "cash"]:
    print(word, one_hot(word))

print("\nVectors for 'win free cash':")

for word in ["win", "free", "cash"]:
    print(word, one_hot(word))


# ============================================================
# TASK 2: BAG OF WORDS AND DOCUMENT-TERM MATRIX
# ============================================================

heading("TASK 2: BAG OF WORDS AND DOCUMENT-TERM MATRIX")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

terms = vectorizer.get_feature_names_out()
matrix = X.toarray()

print("Vocabulary:", list(terms))
print("Shape:", X.shape)

print("\nDocument-Term Matrix:")

for i, row in enumerate(matrix):
    print(f"D{i + 1}:", row.tolist())

win_index = list(terms).index("win")

print("\nD1 win count:", matrix[0, win_index])

vector1 = vectorizer.transform(
    ["win free phone"]
).toarray()[0]

vector2 = vectorizer.transform(
    ["phone free win"]
).toarray()[0]

print("\nVector for 'win free phone':", vector1.tolist())
print("Vector for 'phone free win':", vector2.tolist())

print("Are vectors equal:", np.array_equal(vector1, vector2))


# ============================================================
# TASK 3: TOKENS VS VOCABULARY SIZE
# ============================================================

heading("TASK 3: TOKENS VS VOCABULARY SIZE")

analyzer = vectorizer.build_analyzer()

token_counts = [
    len(analyzer(doc)) for doc in documents
]

total_tokens = sum(token_counts)

document_frequency = np.asarray(
    (X > 0).sum(axis=0)
).ravel()

repeated_words = [
    word
    for word, df in zip(terms, document_frequency)
    if df > 1
]

print("Tokens per document:", token_counts)
print("Total tokens:", total_tokens)
print("Vocabulary size:", len(terms))
print("Words appearing in more than one document:",
      repeated_words)


# ============================================================
# TASK 4: PREPROCESSING INSIDE THE VECTORIZER
# ============================================================

heading("TASK 4: PREPROCESSING INSIDE THE VECTORIZER")

D1 = [documents[0]]

# Default
v1 = CountVectorizer()
v1.fit(D1)

print("Default vocabulary:",
      list(v1.get_feature_names_out()))

# Lowercase=False
v2 = CountVectorizer(lowercase=False)
v2.fit(D1)

print("Lowercase=False vocabulary:",
      list(v2.get_feature_names_out()))

# English stop words
v3 = CountVectorizer(stop_words="english")
v3.fit(D1)

print("Stop words vocabulary:",
      list(v3.get_feature_names_out()))

# Apply stop words to all documents
v4 = CountVectorizer(stop_words="english")
v4.fit(documents)

stop_vocabulary = list(v4.get_feature_names_out())

print("\nD1-D5 stop words vocabulary:",
      stop_vocabulary)

print("'not' retained:", "not" in stop_vocabulary)


# ============================================================
# TASK 5: LIMITING THE VOCABULARY
# ============================================================

heading("TASK 5: LIMITING THE VOCABULARY")

# min_df=2
min_vectorizer = CountVectorizer(min_df=2)
min_vectorizer.fit(documents)

# max_features=5
max_vectorizer = CountVectorizer(max_features=5)
max_vectorizer.fit(documents)

print("Full vocabulary size:", len(terms))

print("\nmin_df=2 vocabulary:",
      list(min_vectorizer.get_feature_names_out()))

print("min_df=2 size:",
      len(min_vectorizer.get_feature_names_out()))

print("\nmax_features=5 vocabulary:",
      list(max_vectorizer.get_feature_names_out()))

print("max_features=5 size:",
      len(max_vectorizer.get_feature_names_out()))


# ============================================================
# TASK 6: N-GRAMS
# ============================================================

heading("TASK 6: N-GRAMS")

# Unigrams and bigrams on D5
for ngram in [(1, 1), (1, 2)]:

    ngram_vectorizer = CountVectorizer(
        ngram_range=ngram
    )

    ngram_vectorizer.fit([documents[4]])

    print(f"\nD5 vocabulary for {ngram}:")

    print(
        list(ngram_vectorizer.get_feature_names_out())
    )


# Vocabulary sizes on all documents
for ngram in [(1, 1), (1, 2), (1, 3)]:

    ngram_vectorizer = CountVectorizer(
        ngram_range=ngram
    )

    ngram_vectorizer.fit(documents)

    print(
        f"Corpus {ngram} vocabulary size:",
        len(ngram_vectorizer.get_feature_names_out())
    )


# ============================================================
# TASK 7: TF-IDF
# ============================================================

heading("TASK 7: TF-IDF")

tfidf_vectorizer = TfidfVectorizer()

X_tfidf = tfidf_vectorizer.fit_transform(documents)

tfidf_terms = tfidf_vectorizer.get_feature_names_out()

tfidf_matrix = X_tfidf.toarray()

print("TF-IDF Matrix (rounded to 2 decimals):")

for i, row in enumerate(tfidf_matrix):

    print(
        f"D{i + 1}:",
        np.round(row, 2).tolist()
    )


# Print word counts and TF-IDF for D1 and D2
for doc_index in [0, 1]:

    print(f"\nD{doc_index + 1} Word Counts and TF-IDF:")

    for j, word in enumerate(terms):

        if matrix[doc_index, j] > 0:

            tfidf_index = list(tfidf_terms).index(word)

            count = matrix[doc_index, j]

            weight = tfidf_matrix[doc_index, tfidf_index]

            print(
                f"{word}: count={count}, "
                f"tfidf={weight:.6f}"
            )


# ============================================================
# TASK 8: TF-IDF BY HAND
# ============================================================

heading("TASK 8: TF-IDF BY HAND")

N = len(documents)

D2_tokens = analyzer(documents[1])

total_D2_tokens = len(D2_tokens)

print("D2 tokens:", D2_tokens)
print("D2 total token count:", total_D2_tokens)

for word in ["cash", "today"]:

    count = D2_tokens.count(word)

    word_index = list(terms).index(word)

    df = int(document_frequency[word_index])

    tf = count / total_D2_tokens

    idf = np.log(N / df)

    tfidf = tf * idf

    print(f"\nWord: {word}")
    print("Count:", count)
    print("TF:", round(tf, 6))
    print("DF:", df)
    print("N:", N)
    print("IDF:", round(idf, 6))
    print("TF-IDF:", round(tfidf, 6))


# ============================================================
# TASK 9: SPARSE MATRIX
# ============================================================

heading("TASK 9: SPARSE MATRIX")

total_cells = matrix.size

nonzero_cells = np.count_nonzero(matrix)

zero_cells = total_cells - nonzero_cells

zero_percentage = (
    zero_cells / total_cells
) * 100

print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)

print(
    f"Percentage of zeros: {zero_percentage:.2f}%"
)


# ============================================================
# TASK 10: FIT AND TRANSFORM
# ============================================================

heading("TASK 10: FIT AND TRANSFORM")

# Fit only on training documents D1-D4
train_documents = documents[:4]

fit_vectorizer = CountVectorizer()

X_train = fit_vectorizer.fit_transform(
    train_documents
)

train_vocabulary = list(
    fit_vectorizer.get_feature_names_out()
)

print("Training vocabulary:", train_vocabulary)

# Transform new texts
X_test = fit_vectorizer.transform(new_texts)

print("\nN1 vector:", X_test.toarray()[0].tolist())

print("N2 vector:", X_test.toarray()[1].tolist())

# Find OOV words
for name, text in zip(["N1", "N2"], new_texts):

    tokens = fit_vectorizer.build_analyzer()(text)

    oov_words = [
        word
        for word in tokens
        if word not in train_vocabulary
    ]

    print(f"\n{name} tokens:", tokens)

    print(f"{name} OOV words:", oov_words)


# ============================================================
# TASK 11: DATA LEAKAGE
# ============================================================

heading("TASK 11: DATA LEAKAGE")

# Fit on training data only
train_only_vectorizer = CountVectorizer()

train_only_vectorizer.fit(train_documents)

train_only_vocabulary = set(
    train_only_vectorizer.get_feature_names_out()
)

# Fit on training and test data
combined_vectorizer = CountVectorizer()

combined_vectorizer.fit(
    train_documents + new_texts
)

combined_vocabulary = set(
    combined_vectorizer.get_feature_names_out()
)

leaked_words = sorted(
    combined_vocabulary - train_only_vocabulary
)

print(
    "Training-only vocabulary:",
    sorted(train_only_vocabulary)
)

print(
    "\nCombined vocabulary:",
    sorted(combined_vocabulary)
)

print("\nWords leaked from test set:", leaked_words)


# ============================================================
# TASK 12: NO MEANING
# ============================================================

heading("TASK 12: NO MEANING")

review_vectorizer = CountVectorizer()

X_reviews = review_vectorizer.fit_transform(reviews)

review_vocabulary = (
    review_vectorizer.get_feature_names_out()
)

print("Review vocabulary:",
      list(review_vocabulary))

print("\nReview vectors:")

for i, row in enumerate(X_reviews.toarray()):

    print(f"R{i + 1}:", row.tolist())


# Cosine Similarity
similarity_matrix = cosine_similarity(X_reviews)

print("\nCosine Similarity Matrix:")

for row in similarity_matrix:

    print(np.round(row, 2).tolist())


print(
    "\nR1-R2 similarity:",
    round(float(similarity_matrix[0, 1]), 2)
)

print(
    "R1-R3 similarity:",
    round(float(similarity_matrix[0, 2]), 2)
)


# ============================================================
# END OF ASSIGNMENT
# ============================================================

