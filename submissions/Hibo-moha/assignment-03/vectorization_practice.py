
# Assignment 3 - Vectorization
# Task 1: One-Hot Encoding

# Step 1: Define vocabulary
vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

print("=" * 50)
print("TASK 1: ONE-HOT ENCODING")
print("=" * 50)

# Step 2: Build a one-hot encoder
def one_hot_encode(word, vocabulary):

    vector = [0] * len(vocabulary)

    if word in vocabulary:
        index = vocabulary.index(word)
        vector[index] = 1

    return vector


# Step 3: Print vector for each word in "free cash"
sentence = "free cash"
words = sentence.split()

print("\nOne-hot vectors for 'free cash':")

for word in words:
    print(word, ":", one_hot_encode(word, vocabulary))


# Step 4: Print vectors for "win free cash"
sentence = "win free cash"
words = sentence.split()

print("\nOne-hot vectors for 'win free cash':")

for word in words:
    print(word, ":", one_hot_encode(word, vocabulary))


# ==========================================
# TASK 2: BAG OF WORDS
# ==========================================

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

print("\n" + "=" * 50)
print("TASK 2: BAG OF WORDS")
print("=" * 50)

# Step 1: Define training documents
documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

# Step 2: Create CountVectorizer
vectorizer = CountVectorizer()

# Step 3: Fit and transform documents
count_matrix = vectorizer.fit_transform(documents)

# Step 4: Print vocabulary
print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

# Step 5: Print vocabulary with indexes
print("\nVocabulary with indexes:")
print(vectorizer.vocabulary_)

# Step 6: Print matrix shape
print("\nDocument-Term Matrix Shape:")
print(count_matrix.shape)

# Step 7: Print full document-term matrix
print("\nDocument-Term Matrix:")
print(count_matrix.toarray())

# Step 8: Transform two sentences
new_sentences = [
    "win free phone",
    "phone free win"
]

new_vectors = vectorizer.transform(new_sentences)

print("\nNew Sentence Vectors:")
for sentence, vector in zip(new_sentences, new_vectors.toarray()):
    print(sentence, ":", vector)


# ==========================================
# TASK 3: TOKENS VS VOCABULARY SIZE
# ==========================================

print("\n" + "=" * 50)
print("TASK 3: TOKENS VS VOCABULARY SIZE")
print("=" * 50)

# Step 1: Count total word occurrences
total_tokens = sum(len(doc.split()) for doc in documents)

# Step 2: Get vocabulary size
vocabulary_size = len(vectorizer.get_feature_names_out())

print("\nTotal Tokens:")
print(total_tokens)

print("\nVocabulary Size:")
print(vocabulary_size)

# Step 3: Find words appearing in more than one document
document_frequency = np.asarray(
    (count_matrix > 0).sum(axis=0)
).flatten()

words = vectorizer.get_feature_names_out()

repeated_words = [
    words[i]
    for i in range(len(words))
    if document_frequency[i] > 1
]

print("\nWords appearing in more than one document:")
print(repeated_words)


# ==========================================
# TASK 4: PREPROCESSING INSIDE VECTORIZER
# ==========================================

print("\n" + "=" * 50)
print("TASK 4: PREPROCESSING INSIDE VECTORIZER")
print("=" * 50)

# Use D1 only
D1 = documents[0]

# Step 1: Default vectorizer
default_vectorizer = CountVectorizer()
default_vectorizer.fit([D1])

print("\n1. Default Vocabulary:")
print(default_vectorizer.get_feature_names_out())


# Step 2: lowercase=False
case_vectorizer = CountVectorizer(lowercase=False)
case_vectorizer.fit([D1])

print("\n2. Vocabulary with lowercase=False:")
print(case_vectorizer.get_feature_names_out())


# Step 3: Remove English stopwords
stop_vectorizer = CountVectorizer(stop_words="english")
stop_vectorizer.fit([D1])

print("\n3. Vocabulary with stop_words='english':")
print(stop_vectorizer.get_feature_names_out())


# Step 4: Check stopwords on all documents
all_stop_vectorizer = CountVectorizer(stop_words="english")
all_stop_vectorizer.fit(documents)

print("\n4. Vocabulary with stopwords on D1-D5:")
print(all_stop_vectorizer.get_feature_names_out())

print("\nIs 'not' in vocabulary?")
print("not" in all_stop_vectorizer.vocabulary_)


# ==========================================
# ADDITIONAL IMPORTS
# ==========================================

import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# TASK 5: LIMITING THE VOCABULARY
# ==========================================

print("\n" + "=" * 50)
print("TASK 5: LIMITING THE VOCABULARY")
print("=" * 50)

# min_df = 2
min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

print("\nVocabulary with min_df=2:")
print(min_df_vectorizer.get_feature_names_out())
print("Vocabulary size:", len(min_df_vectorizer.get_feature_names_out()))

# max_features = 5
max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

print("\nVocabulary with max_features=5:")
print(max_features_vectorizer.get_feature_names_out())
print("Vocabulary size:", len(max_features_vectorizer.get_feature_names_out()))

print("\nFull vocabulary size:", len(vectorizer.get_feature_names_out()))


# ==========================================
# TASK 6: N-GRAMS
# ==========================================

print("\n" + "=" * 50)
print("TASK 6: N-GRAMS")
print("=" * 50)

D5 = documents[4]

# D5: Unigrams
unigram_D5 = CountVectorizer(ngram_range=(1, 1))
unigram_D5.fit([D5])

print("\nD5 Unigram Vocabulary:")
print(unigram_D5.get_feature_names_out())

# D5: Unigrams and bigrams
bigram_D5 = CountVectorizer(ngram_range=(1, 2))
bigram_D5.fit([D5])

print("\nD5 Unigram + Bigram Vocabulary:")
print(bigram_D5.get_feature_names_out())

# Compare vocabulary sizes on D1-D5
for n in [(1, 1), (1, 2), (1, 3)]:
    ngram_vectorizer = CountVectorizer(ngram_range=n)
    ngram_vectorizer.fit(documents)

    print(f"\nN-gram range {n}:")
    print("Vocabulary size:", len(ngram_vectorizer.get_feature_names_out()))


# ==========================================
# TASK 7: TF-IDF
# ==========================================

print("\n" + "=" * 50)
print("TASK 7: TF-IDF")
print("=" * 50)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_words = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF Matrix (rounded to 2 decimals):")
print(np.round(tfidf_matrix.toarray(), 2))

# Print counts and TF-IDF weights for D1 and D2
for doc_index, doc_name in [(0, "D1"), (1, "D2")]:

    print(f"\n{doc_name}: Word Counts and TF-IDF Weights")

    for word_index, word in enumerate(tfidf_words):

        count = count_matrix[doc_index, vectorizer.vocabulary_[word]] \
            if word in vectorizer.vocabulary_ else 0

        weight = tfidf_matrix[doc_index, word_index]

        if count > 0:
            print(f"{word:12} Count: {count} | TF-IDF: {weight:.4f}")


# ==========================================
# TASK 8: TF-IDF BY HAND
# ==========================================

print("\n" + "=" * 50)
print("TASK 8: TF-IDF BY HAND")
print("=" * 50)

N = len(documents)
D2 = documents[1]

# CountVectorizer tokenization for D2
D2_tokens = vectorizer.build_analyzer()(D2)
total_D2_tokens = len(D2_tokens)

print("\nD2 Tokens:", D2_tokens)
print("Total D2 Tokens:", total_D2_tokens)
print("Total Documents:", N)

for word in ["cash", "today"]:

    count = D2_tokens.count(word)

    # Document frequency
    df = int(document_frequency[vectorizer.vocabulary_[word]])

    # Textbook TF
    tf = count / total_D2_tokens

    # Textbook IDF
    idf = math.log(N / df)

    # Textbook TF-IDF
    tfidf_hand = tf * idf

    print(f"\nWord: {word}")
    print("Count:", count)
    print("Document Frequency:", df)
    print("TF:", round(tf, 4))
    print("IDF:", round(idf, 4))
    print("TF-IDF:", round(tfidf_hand, 4))


# ==========================================
# TASK 9: SPARSE MATRIX
# ==========================================

print("\n" + "=" * 50)
print("TASK 9: SPARSE MATRIX")
print("=" * 50)

total_cells = count_matrix.shape[0] * count_matrix.shape[1]
non_zero_cells = count_matrix.nnz
zero_cells = total_cells - non_zero_cells
zero_percentage = (zero_cells / total_cells) * 100

print("\nTotal Cells:", total_cells)
print("Non-zero Cells:", non_zero_cells)
print("Zero Cells:", zero_cells)
print("Percentage of Zeros:", round(zero_percentage, 2), "%")


# ==========================================
# TASK 10: FIT AND TRANSFORM
# ==========================================

print("\n" + "=" * 50)
print("TASK 10: FIT AND TRANSFORM")
print("=" * 50)

training_documents = documents[:4]

test_documents = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]

training_vectorizer = CountVectorizer()

# Fit only on training data
training_vectorizer.fit(training_documents)

# Transform test data
test_matrix = training_vectorizer.transform(test_documents)

print("\nTraining Vocabulary:")
print(training_vectorizer.get_feature_names_out())

print("\nTest Vectors:")
for text, vector in zip(test_documents, test_matrix.toarray()):
    print(text)
    print(vector)

# Find out-of-vocabulary words
analyzer = training_vectorizer.build_analyzer()
training_vocabulary = training_vectorizer.vocabulary_

for text in test_documents:
    tokens = analyzer(text)

    oov_words = [
        word for word in tokens
        if word not in training_vocabulary
    ]

    print("\nText:", text)
    print("Out-of-Vocabulary Words:", oov_words)


# ==========================================
# TASK 11: DATA LEAKAGE
# ==========================================

print("\n" + "=" * 50)
print("TASK 11: DATA LEAKAGE")
print("=" * 50)

# Correct: Fit only on training documents
train_only_vectorizer = CountVectorizer()
train_only_vectorizer.fit(training_documents)

# Incorrect: Fit on training + test documents
leaked_vectorizer = CountVectorizer()
leaked_vectorizer.fit(training_documents + test_documents)

train_vocab = set(train_only_vectorizer.get_feature_names_out())
leaked_vocab = set(leaked_vectorizer.get_feature_names_out())

print("\nTraining-only Vocabulary:")
print(sorted(train_vocab))

print("\nTraining + Test Vocabulary:")
print(sorted(leaked_vocab))

print("\nWords Leaked from Test Set:")
print(sorted(leaked_vocab - train_vocab))


# ==========================================
# TASK 12: BAG OF WORDS DOES NOT CAPTURE MEANING
# ==========================================

print("\n" + "=" * 50)
print("TASK 12: BAG OF WORDS AND MEANING")
print("=" * 50)

reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]

review_vectorizer = CountVectorizer()

review_matrix = review_vectorizer.fit_transform(reviews)

print("\nReview Vocabulary:")
print(review_vectorizer.get_feature_names_out())

print("\nReview Vectors:")
print(review_matrix.toarray())

# Cosine similarity
similarity_matrix = cosine_similarity(review_matrix)

print("\nCosine Similarity Matrix:")
print(np.round(similarity_matrix, 4))

print("\nReview Labels:")
print("R1: the food was tasty")
print("R2: the food was delicious")
print("R3: the food was terrible")

print("\nR1 vs R2:", round(similarity_matrix[0, 1], 4))
print("R1 vs R3:", round(similarity_matrix[0, 2], 4))
print("R2 vs R3:", round(similarity_matrix[1, 2], 4))

print("\nEND OF ASSIGNMENT 3 PRACTICAL")
