import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


np.set_printoptions(linewidth=200)


# ==================================================
# DATASETS
# ==================================================

documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

# New Texts
new_texts = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]

# Reviews
reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]


# ==================================================
# TASK 1 - ONE-HOT ENCODING
# ==================================================

print("\n1) ======= ONE-HOT ENCODING ======")

one_hot_vectorization = ["cash", "class", "free", "lunch", "notes", "win"]

print("\n=== Vocabulary ===")
print(one_hot_vectorization)


def one_hot_encode(word):
    vector = []

    for vocab_word in one_hot_vectorization:
        if word == vocab_word:
            vector.append(1)
        else:
            vector.append(0)

    return vector


print("\n=== One-Hot Encoded Words ===")
print("free:", one_hot_encode("free"))
print("cash:", one_hot_encode("cash"))
print("win:", one_hot_encode("win"))


print("\n=== One-Hot Encoding: free cash ===")

for word in "free cash".split():
    print(word, one_hot_encode(word))


print("\n=== One-Hot Encoding: win free cash ===")

for word in "win free cash".split():
    print(word, one_hot_encode(word))


# ==================================================
# TASK 2 - BAG OF WORDS
# ==================================================

print("\n\n2) ======= BAG OF WORDS ======")

count_vectorizer = CountVectorizer()

X = count_vectorizer.fit_transform(documents)

print("\n=== Vocabulary ===")
print(count_vectorizer.get_feature_names_out().tolist())

print("\n=== Matrix Shape ===")
print(X.shape)

print("\n=== Document-Term Matrix ===")
print(X.toarray())


# Test Vectors

test_texts = [
    "win free phone",
    "phone free win"
]

test_vector = count_vectorizer.transform(test_texts)

print("\n=== Test Vectors ===")
print(test_vector.toarray())


# ==================================================
# TASK 3 - TOKENS VS VOCABULARY
# ==================================================

print("\n\n3) ======= TOKENS VS VOCABULARY ======")

document_freq = np.asarray((X > 0).sum(axis=0)).ravel()

words = count_vectorizer.get_feature_names_out()

print("\n=== Words Appearing in More Than One Document ===")

for word, frequency in zip(words, document_freq):
    if frequency > 1:
        print(word, frequency)


total_tokens = X.sum()
total_vocabulary = len(words)

print("\nTotal Tokens:", total_tokens)
print("Total Vocabulary:", total_vocabulary)


# ==================================================
# TASK 4 - LOWERCASE AND STOPWORDS
# ==================================================

print("\n\n4) ======= LOWERCASE AND STOPWORDS ======")

# Default settings

default_vectorizer = CountVectorizer()
default_vectorizer.fit([documents[0]])

print("\n=== Default Vocabulary ===")
print(default_vectorizer.get_feature_names_out().tolist())


# Disable lowercase conversion

case_vectorizer = CountVectorizer(lowercase=False)
case_vectorizer.fit([documents[0]])

print("\n=== Vocabulary with lowercase=False ===")
print(case_vectorizer.get_feature_names_out().tolist())


# Remove English stopwords

stop_vectorizer = CountVectorizer(stop_words="english")
stop_vectorizer.fit([documents[0]])

print("\n=== Vocabulary with English Stopwords ===")
print(stop_vectorizer.get_feature_names_out().tolist())


# Check whether 'not' is removed

all_stop_vectorizer = CountVectorizer(stop_words="english")
all_stop_vectorizer.fit(documents)

print("\n=== Is 'not' in Vocabulary? ===")
print("not" in all_stop_vectorizer.vocabulary_)


# ==================================================
# TASK 5 - min_df AND max_features
# ==================================================

print("\n\n5) ======= VOCABULARY LIMITS ======")

# Keep words appearing in at least 2 documents

min_df_vectorizer = CountVectorizer(min_df=2)

min_df_matrix = min_df_vectorizer.fit_transform(documents)

print("\n=== Vocabulary with min_df=2 ===")
print(min_df_vectorizer.get_feature_names_out().tolist())

print("Vocabulary Size:", len(min_df_vectorizer.get_feature_names_out()))


# Keep only 5 most frequent words

max_features_vectorizer = CountVectorizer(max_features=5)

max_features_matrix = max_features_vectorizer.fit_transform(documents)

print("\n=== Vocabulary with max_features=5 ===")
print(max_features_vectorizer.get_feature_names_out().tolist())

print("Vocabulary Size:", len(max_features_vectorizer.get_feature_names_out()))


# ==================================================
# TASK 6 - N-GRAMS
# ==================================================

print("\n\n6) ======= N-GRAMS ======")

# Use D5 only

d5 = [documents[4]]


# Unigrams

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit(d5)

print("\n=== D5 Unigrams ===")
print(unigram_vectorizer.get_feature_names_out().tolist())


# Unigrams and Bigrams

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit(d5)

print("\n=== D5 Unigrams and Bigrams ===")
print(bigram_vectorizer.get_feature_names_out().tolist())


# Compare vocabulary sizes across all documents

for ngram in [(1, 1), (1, 2), (1, 3)]:

    ngram_vectorizer = CountVectorizer(ngram_range=ngram)
    ngram_vectorizer.fit(documents)

    print(
        f"\nN-gram range {ngram}:",
        len(ngram_vectorizer.get_feature_names_out()),
        "features"
    )


# ==================================================
# TASK 7 - TF-IDF
# ==================================================

print("\n\n7) ======= TF-IDF ======")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_words = tfidf_vectorizer.get_feature_names_out()

print("\n=== TF-IDF Matrix ===")
print(np.round(tfidf_matrix.toarray(), 2))


# Print word, count and TF-IDF weight for D1 and D2

for doc_index in [0, 1]:

    print(f"\n=== Document D{doc_index + 1} ===")

    for word in tfidf_words:

        count_index = count_vectorizer.vocabulary_[word]
        tfidf_index = tfidf_vectorizer.vocabulary_[word]

        count = X[doc_index, count_index]
        weight = tfidf_matrix[doc_index, tfidf_index]

        if count > 0:
            print(
                word,
                "Count:", count,
                "TF-IDF:", round(float(weight), 2)
            )


# ==================================================
# TASK 8 - MANUAL TF-IDF CALCULATION
# ==================================================

print("\n\n8) ======= MANUAL TF-IDF CALCULATION ======")

N = len(documents)

# Total tokens in D2

d2_total_tokens = X[1].sum()

for word in ["cash", "today"]:

    word_index = count_vectorizer.vocabulary_[word]

    # Term Frequency

    count = X[1, word_index]
    tf = count / d2_total_tokens

    # Document Frequency

    df = int(document_freq[word_index])

    # Textbook IDF formula

    idf = np.log(N / df)

    # TF-IDF

    tfidf_score = tf * idf

    print(f"\nWord: {word}")
    print("Count:", count)
    print("TF:", round(float(tf), 4))
    print("DF:", df)
    print("IDF:", round(float(idf), 4))
    print("TF-IDF:", round(float(tfidf_score), 4))


# ==================================================
# TASK 9 - MATRIX SPARSITY
# ==================================================

print("\n\n9) ======= MATRIX SPARSITY ======")

total_cells = X.shape[0] * X.shape[1]

nonzero_cells = X.count_nonzero()

zero_cells = total_cells - nonzero_cells

zero_percentage = (zero_cells / total_cells) * 100

print("Total Cells:", total_cells)
print("Non-zero Cells:", nonzero_cells)
print("Zero Cells:", zero_cells)
print("Percentage of Zeros:", round(zero_percentage, 2), "%")


# ==================================================
# TASK 10 - OUT-OF-VOCABULARY (OOV)
# ==================================================

print("\n\n10) ======= OUT-OF-VOCABULARY ======")

# Training documents: D1-D4 only

train_documents = documents[:4]

oov_vectorizer = CountVectorizer()

oov_matrix = oov_vectorizer.fit_transform(train_documents)

print("\n=== Training Vocabulary ===")
print(oov_vectorizer.get_feature_names_out().tolist())


# Transform new texts

new_vectors = oov_vectorizer.transform(new_texts)

print("\n=== New Text Vectors ===")
print(new_vectors.toarray())


# Find OOV words

analyzer = oov_vectorizer.build_analyzer()

known_words = set(oov_vectorizer.get_feature_names_out())

for text in new_texts:

    tokens = analyzer(text)

    oov_words = [
        word for word in tokens
        if word not in known_words
    ]

    print("\nText:", text)
    print("OOV Words:", oov_words)


# ==================================================
# TASK 11 - DATA LEAKAGE
# ==================================================

print("\n\n11) ======= DATA LEAKAGE ======")

# Fit using training documents only

train_only_vectorizer = CountVectorizer()

train_only_vectorizer.fit(documents[:4])

train_vocabulary = set(
    train_only_vectorizer.get_feature_names_out()
)

print("\n=== Training Vocabulary ===")
print(sorted(train_vocabulary))


# Incorrect approach: include test texts during fitting

combined_vectorizer = CountVectorizer()

combined_vectorizer.fit(documents[:4] + new_texts)

combined_vocabulary = set(
    combined_vectorizer.get_feature_names_out()
)

print("\n=== Combined Vocabulary ===")
print(sorted(combined_vocabulary))


# Words introduced by test data

leaked_words = combined_vocabulary - train_vocabulary

print("\n=== Words Introduced by Test Data ===")
print(sorted(leaked_words))


# ==================================================
# TASK 12 - COSINE SIMILARITY
# ==================================================

print("\n\n12) ======= COSINE SIMILARITY ======")

review_vectorizer = CountVectorizer()

review_matrix = review_vectorizer.fit_transform(reviews)

print("\n=== Review Vocabulary ===")
print(review_vectorizer.get_feature_names_out().tolist())


print("\n=== Review Vectors ===")
print(review_matrix.toarray())


# Calculate pairwise cosine similarity

similarity_matrix = cosine_similarity(review_matrix)

print("\n=== Cosine Similarity Matrix ===")
print(np.round(similarity_matrix, 2))


# ==================================================
# END OF ASSIGNMENT
# ==================================================

print("\n\n======= ASSIGNMENT 3 COMPLETED =======")
