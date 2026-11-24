
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


np.set_printoptions(linewidth=200, precision=3, suppress=True)


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


# ==========================================
# 1. ONE-HOT ENCODING
# ==========================================

print("\n=== 1. ONE-HOT ENCODING ===")

one_hot_vocabulary = [
    "cash", "class", "free", "lunch", "notes", "win"
]

print("Vocabulary:", one_hot_vocabulary)

def one_hot_encode(word):
    return [
        1 if vocabulary_word == word else 0
        for vocabulary_word in one_hot_vocabulary
    ]

for word in "free cash".split():
    print(word, "->", one_hot_encode(word))

sentence = "win free cash"

print("\nSentence:", sentence)

for word in sentence.split():
    print(word, "->", one_hot_encode(word))


# ==========================================
# 2. BAG OF WORDS
# ==========================================

print("\n=== 2. BAG OF WORDS ===")

count_vectorizer = CountVectorizer()

count_matrix = count_vectorizer.fit_transform(documents)

vocabulary = count_vectorizer.get_feature_names_out()

print("Vocabulary:", vocabulary.tolist())
print("Shape:", count_matrix.shape)
print("Document-Term Matrix:")
print(count_matrix.toarray())

d1_counts = count_matrix.toarray()[0]

print("\nD1 words with count 2:")

for word, count in zip(vocabulary, d1_counts):
    if count == 2:
        print(word, "->", count)

vector1 = count_vectorizer.transform(
    ["win free phone"]
).toarray()

vector2 = count_vectorizer.transform(
    ["phone free win"]
).toarray()

print("\nwin free phone:", vector1)
print("phone free win:", vector2)
print("Vectors equal:", np.array_equal(vector1, vector2))


# ==========================================
# 3. TOKENS VS VOCABULARY SIZE
# ==========================================

print("\n=== 3. TOKENS VS VOCABULARY SIZE ===")

total_tokens = sum(
    len(CountVectorizer().build_analyzer()(doc))
    for doc in documents
)

vocabulary_size = len(vocabulary)

print("Total tokens:", total_tokens)
print("Vocabulary size:", vocabulary_size)

document_frequency = np.count_nonzero(
    count_matrix.toarray(), axis=0
)

print("\nWords appearing in more than one document:")

for word, df in zip(vocabulary, document_frequency):
    if df > 1:
        print(word, "->", df)


# ==========================================
# 4. PREPROCESSING INSIDE THE VECTORIZER
# ==========================================

print("\n=== 4. PREPROCESSING INSIDE THE VECTORIZER ===")

d1 = [documents[0]]

default_vectorizer = CountVectorizer()
default_vectorizer.fit(d1)

print("Default:", default_vectorizer.get_feature_names_out().tolist())

case_vectorizer = CountVectorizer(lowercase=False)
case_vectorizer.fit(d1)

print("lowercase=False:", case_vectorizer.get_feature_names_out().tolist())

stop_vectorizer = CountVectorizer(stop_words="english")
stop_vectorizer.fit(d1)

print("stop_words='english':", stop_vectorizer.get_feature_names_out().tolist())

stop_all_vectorizer = CountVectorizer(stop_words="english")
stop_all_vectorizer.fit(documents)

stop_all_vocabulary = stop_all_vectorizer.get_feature_names_out()

print("\nIs 'not' in vocabulary:", "not" in stop_all_vocabulary)


# ==========================================
# 5. LIMITING THE VOCABULARY
# ==========================================

print("\n=== 5. LIMITING THE VOCABULARY ===")

full_size = len(vocabulary)

min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

min_df_vocabulary = min_df_vectorizer.get_feature_names_out()

print("min_df=2:", min_df_vocabulary.tolist())
print("min_df size:", len(min_df_vocabulary))

max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

max_features_vocabulary = max_features_vectorizer.get_feature_names_out()

print("max_features=5:", max_features_vocabulary.tolist())
print("max_features size:", len(max_features_vocabulary))

print("Full vocabulary size:", full_size)


# ==========================================
# 6. N-GRAMS
# ==========================================

print("\n=== 6. N-GRAMS ===")

d5 = [documents[4]]

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit(d5)

print("Unigrams:", unigram_vectorizer.get_feature_names_out().tolist())

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit(d5)

print("Unigrams and bigrams:", bigram_vectorizer.get_feature_names_out().tolist())

for n in [1, 2, 3]:

    ngram_vectorizer = CountVectorizer(
        ngram_range=(1, n)
    )

    ngram_vectorizer.fit(documents)

    size = len(ngram_vectorizer.get_feature_names_out())

    print(f"(1, {n}) vocabulary size:", size)


# ==========================================
# 7. TF-IDF
# ==========================================

print("\n=== 7. TF-IDF ===")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF Matrix:")
print(tfidf_matrix.toarray().round(2))

for doc_index in [0, 1]:

    print(f"\nD{doc_index + 1}:")

    counts = count_matrix.toarray()[doc_index]

    weights = tfidf_matrix.toarray()[doc_index]

    for word, count, weight in zip(
        vocabulary, counts, weights
    ):

        if count > 0:
            print(
                word,
                "| Count:", count,
                "| TF-IDF:", round(weight, 3)
            )


# ==========================================
# 8. TF-IDF BY HAND
# ==========================================

print("\n=== 8. TF-IDF BY HAND ===")

N = len(documents)

d2_index = 1

words_in_d2 = count_matrix[d2_index].sum()

print("Document:", documents[d2_index])
print("Total tokens:", words_in_d2)

for word in ["cash", "today"]:

    word_index = list(vocabulary).index(word)

    count = count_matrix[d2_index, word_index]

    df = np.count_nonzero(
        count_matrix[:, word_index].toarray()
    )

    tf = count / words_in_d2

    idf = np.log(N / df)

    tfidf_hand = tf * idf

    print(f"\nWord: {word}")
    print("Count:", count)
    print("DF:", df)
    print("TF:", round(tf, 4))
    print("IDF:", round(idf, 4))
    print("TF-IDF:", round(tfidf_hand, 4))

    sklearn_weight = tfidf_matrix[d2_index, word_index]

    print("Scikit-learn TF-IDF:", round(float(sklearn_weight), 4))


# ==========================================
# 9. SPARSE MATRIX
# ==========================================

print("\n=== 9. SPARSE MATRIX ===")

total_cells = count_matrix.shape[0] * count_matrix.shape[1]

non_zero_cells = count_matrix.nnz

zero_cells = total_cells - non_zero_cells

zero_percentage = (zero_cells / total_cells) * 100

print("Total cells:", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(zero_percentage, 2), "%")


# ==========================================
# 10. FIT AND TRANSFORM
# ==========================================

print("\n=== 10. FIT AND TRANSFORM ===")

train_documents = documents[:4]

fit_vectorizer = CountVectorizer()

X_train = fit_vectorizer.fit_transform(train_documents)

train_vocabulary = fit_vectorizer.get_feature_names_out()

print("Training vocabulary:", train_vocabulary.tolist())

print("Training matrix:")
print(X_train.toarray())

new_matrix = fit_vectorizer.transform(new_texts)

print("\nNew text vectors:")

for text, vector in zip(new_texts, new_matrix.toarray()):
    print("\nText:", text)
    print("Vector:", vector)

known_words = set(train_vocabulary)

print("\nOut-of-vocabulary words:")

for text in new_texts:

    tokens = CountVectorizer().build_analyzer()(text)

    oov_words = sorted(set(tokens) - known_words)

    print("\nText:", text)
    print("OOV words:", oov_words)


# ==========================================
# 11. DATA LEAKAGE
# ==========================================

print("\n=== 11. DATA LEAKAGE ===")

train_only_vectorizer = CountVectorizer()

train_only_vectorizer.fit(train_documents)

train_only_vocabulary = set(
    train_only_vectorizer.get_feature_names_out()
)

leaky_vectorizer = CountVectorizer()

leaky_vectorizer.fit(train_documents + new_texts)

leaky_vocabulary = set(
    leaky_vectorizer.get_feature_names_out()
)

print("Training-only vocabulary:")
print(sorted(train_only_vocabulary))

print("\nTraining + test vocabulary:")
print(sorted(leaky_vocabulary))

leaked_words = leaky_vocabulary - train_only_vocabulary

print("\nWords leaked from test set:")
print(sorted(leaked_words))


# ==========================================
# 12. NO MEANING + COSINE SIMILARITY
# ==========================================

print("\n=== 12. NO MEANING + COSINE SIMILARITY ===")

review_vectorizer = CountVectorizer()

review_matrix = review_vectorizer.fit_transform(reviews)

review_vocabulary = review_vectorizer.get_feature_names_out()

print("Vocabulary:")
print(review_vocabulary.tolist())

print("\nReview vectors:")

for review, vector in zip(reviews, review_matrix.toarray()):
    print(review, "->", vector)

similarity_matrix = cosine_similarity(review_matrix)

print("\nCosine Similarity Matrix:")
print(similarity_matrix.round(3))

print("\nR1 vs R2:", round(similarity_matrix[0, 1], 3))

print("R1 vs R3:", round(similarity_matrix[0, 2], 3))


print("\n=== END  ===")
