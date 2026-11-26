# Assignment 3 - Vectorization

# Task 1: One-Hot Encoding

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

print("\n=== TASK 1: ONE-HOT ENCODING ===")

# Create one-hot encoder
def one_hot_encode(word):
    vector = []

    for vocab_word in vocabulary:
        if word == vocab_word:
            vector.append(1)
        else:
            vector.append(0)

    return vector

# Encode each word in "free cash"
words = ["free", "cash"]

for word in words:
    print(word, ":", one_hot_encode(word))

    print("\nEncoding: win free cash")

sentence = ["win", "free", "cash"]

for word in sentence:
    print(word, ":", one_hot_encode(word))

    # TASK 2: BAG OF WORDS

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

print("\n=== TASK 2: BAG OF WORDS ===")

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

print("\nMatrix Shape:")
print(X.shape)

print("\nDocument-Term Matrix:")
print(X.toarray())



# TASK 3: TOKENS VS VOCABULARY SIZE

print("\n=== TASK 3: TOKENS VS VOCABULARY SIZE ===")

# Total number of tokens
total_tokens = sum(len(doc.split()) for doc in documents)

# Vocabulary size
vocabulary_size = len(vectorizer.get_feature_names_out())

print("Total tokens:", total_tokens)
print("Vocabulary size:", vocabulary_size)

# Words appearing in more than one document
import numpy as np

word_counts = (X.toarray() > 0).sum(axis=0)

repeated_words = vectorizer.get_feature_names_out()[word_counts > 1]

print("Words appearing in more than one document:")
print(repeated_words)

# TASK 4: PREPROCESSING INSIDE THE VECTORIZER

print("\n=== TASK 4: PREPROCESSING ===")

D1 = ["Win a FREE phone now! Reply WIN to claim."]

# Default vectorizer
default_vectorizer = CountVectorizer()
default_vectorizer.fit(D1)

print("\nDefault Vocabulary:")
print(default_vectorizer.get_feature_names_out())

# Lowercase False
case_vectorizer = CountVectorizer(lowercase=False)
case_vectorizer.fit(D1)

print("\nLowercase=False Vocabulary:")
print(case_vectorizer.get_feature_names_out())

# Stopwords English
stop_vectorizer = CountVectorizer(stop_words="english")
stop_vectorizer.fit(D1)

print("\nStopwords Vocabulary:")
print(stop_vectorizer.get_feature_names_out())

# Check NOT in all documents
not_vectorizer = CountVectorizer(stop_words="english")
not_vectorizer.fit(documents)

print("\nIs 'not' in vocabulary?")
print("not" in not_vectorizer.vocabulary_)


# TASK 5: LIMITING THE VOCABULARY

print("\n=== TASK 5: LIMITING THE VOCABULARY ===")

# min_df = 2
min_vectorizer = CountVectorizer(min_df=2)
min_vectorizer.fit(documents)

print("\nVocabulary with min_df=2:")
print(min_vectorizer.get_feature_names_out())

print("Vocabulary size:", len(min_vectorizer.get_feature_names_out()))

# max_features = 5
max_vectorizer = CountVectorizer(max_features=5)
max_vectorizer.fit(documents)

print("\nVocabulary with max_features=5:")
print(max_vectorizer.get_feature_names_out())

print("Vocabulary size:", len(max_vectorizer.get_feature_names_out()))

# Full vocabulary size
print("\nFull vocabulary size:", len(vectorizer.get_feature_names_out()))

# TASK 6: N-GRAMS

print("\n=== TASK 6: N-GRAMS ===")

D5 = ["The lecture was not good, but the notes were great."]

# Unigrams
uni_vectorizer = CountVectorizer(ngram_range=(1, 1))
uni_vectorizer.fit(D5)

print("\nUnigram Vocabulary:")
print(uni_vectorizer.get_feature_names_out())

# Unigrams and Bigrams
bi_vectorizer = CountVectorizer(ngram_range=(1, 2))
bi_vectorizer.fit(D5)

print("\nUnigram + Bigram Vocabulary:")
print(bi_vectorizer.get_feature_names_out())

# Vocabulary growth on all documents
for n in [(1, 1), (1, 2), (1, 3)]:
    ngram_vectorizer = CountVectorizer(ngram_range=n)
    ngram_vectorizer.fit(documents)

    print(
        f"\nN-gram range {n}:",
        len(ngram_vectorizer.get_feature_names_out())
    )

    # TASK 7: TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

print("\n=== TASK 7: TF-IDF ===")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF Matrix:")
print(np.round(tfidf_matrix.toarray(), 2))

# D1 and D2: Count and TF-IDF weights
count_matrix = vectorizer.transform(documents)

for doc_index in [0, 1]:
    print(f"\nDocument D{doc_index + 1}:")

    counts = count_matrix.toarray()[doc_index]
    weights = tfidf_matrix.toarray()[doc_index]

    for word, count, weight in zip(feature_names, counts, weights):
        if count > 0:
            print(word, "| Count:", count,
                  "| TF-IDF:", round(weight, 2))


  # TASK 8: TF-IDF BY HAND

import math

print("\n=== TASK 8: TF-IDF BY HAND ===")

N = len(documents)

D2_words = documents[1].lower().replace(".", "").replace(":", "").split()

for word in ["cash", "today"]:

    tf = D2_words.count(word)

    df = sum(
        1 for doc in documents
        if word in CountVectorizer().build_analyzer()(doc)
    )

    idf = math.log(N / df)

    tfidf = tf * idf

    print(f"\nWord: {word}")
    print("TF:", tf)
    print("DF:", df)
    print("IDF:", round(idf, 4))
    print("TF-IDF:", round(tfidf, 4))          



 # TASK 9: SPARSE MATRIX

print("\n=== TASK 9: SPARSE MATRIX ===")

matrix = X.toarray()

total_cells = matrix.size
non_zero_cells = np.count_nonzero(matrix)
zero_cells = total_cells - non_zero_cells

zero_percentage = (zero_cells / total_cells) * 100

print("Total cells:", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(zero_percentage, 2), "%")   



# TASK 10: OUT-OF-VOCABULARY WORDS

print("\n=== TASK 10: OOV WORDS ===")

train_docs = documents[:4]

new_texts = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]

train_vectorizer = CountVectorizer()
train_matrix = train_vectorizer.fit_transform(train_docs)

new_matrix = train_vectorizer.transform(new_texts)

print("Training vocabulary:")
print(train_vectorizer.get_feature_names_out())

print("\nNew text vectors:")
print(new_matrix.toarray())

analyzer = train_vectorizer.build_analyzer()

for text in new_texts:
    tokens = analyzer(text)
    oov_words = [
        word for word in tokens
        if word not in train_vectorizer.vocabulary_
    ]

    print("\nText:", text)
    print("OOV words:", oov_words)

    # TASK 11: DATA LEAKAGE

print("\n=== TASK 11: DATA LEAKAGE ===")

train_only = CountVectorizer()
train_only.fit(documents[:4])

train_and_test = CountVectorizer()
train_and_test.fit(documents[:4] + new_texts)

print("Training vocabulary:")
print(train_only.get_feature_names_out())

print("\nVocabulary after adding test data:")
print(train_and_test.get_feature_names_out())

new_words = set(train_and_test.vocabulary_) - set(train_only.vocabulary_)

print("\nWords added from test data:")
print(sorted(new_words))

# TASK 12: BAG OF WORDS AND COSINE SIMILARITY

print("\n=== TASK 12: COSINE SIMILARITY ===")

from sklearn.metrics.pairwise import cosine_similarity

reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]

review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(reviews)

similarity = cosine_similarity(review_matrix)

print("Vocabulary:")
print(review_vectorizer.get_feature_names_out())

print("\nDocument-Term Matrix:")
print(review_matrix.toarray())

print("\nCosine Similarity:")
print(np.round(similarity, 2))