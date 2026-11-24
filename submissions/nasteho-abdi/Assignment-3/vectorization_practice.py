import re
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# ASSIGNMENT 3 - TEXT VECTORIZATION
# =====================================================

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


def print_matrix(matrix, vocabulary, names):
    print("Vocabulary:", list(vocabulary))
    print("Shape:", matrix.shape)
    print("Document-Term Matrix:")
    print("Documents:", list(vocabulary))
    for name, row in zip(names, matrix.toarray()):
        print(name, row)


# =====================================================
# TASK 1 - ONE HOT ENCODING
# =====================================================

heading("TASK 1 - ONE HOT ENCODING")

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

one_hot = {}

for i, word in enumerate(vocabulary):
    vector = [0] * len(vocabulary)
    vector[i] = 1
    one_hot[word] = vector

print("Vocabulary:", vocabulary)

print("\nOne-hot vectors for 'free cash':")

for word in ["free", "cash"]:
    print(word, ":", one_hot[word])

sentence = ["win", "free", "cash"]

print("\nOne-hot vectors for 'win free cash':")

for word in sentence:
    print(word, ":", one_hot[word])

print("\nNumber of vectors:", len(sentence))
print("Vector length:", len(vocabulary))


# =====================================================
# TASK 2 - BAG OF WORDS
# =====================================================

heading("TASK 2 - BAG OF WORDS")

count_vectorizer = CountVectorizer()

count_matrix = count_vectorizer.fit_transform(documents)

feature_names = count_vectorizer.get_feature_names_out()

print("Vocabulary:")
print(list(feature_names))

print("\nMatrix shape:", count_matrix.shape)

print("\nFull Document-Term Matrix:")

for i, row in enumerate(count_matrix.toarray()):
    print(f"D{i+1}:", row)

print("\nCount of WIN in D1:")

win_index = list(feature_names).index("win")
print(count_matrix.toarray()[0, win_index])

print("\nTransforming two sentences:")

test_sentences = [
    "win free phone",
    "phone free win"
]

test_matrix = count_vectorizer.transform(test_sentences)

for sentence, vector in zip(test_sentences, test_matrix.toarray()):
    print(sentence, ":", vector)

print("Are the vectors equal?",
      np.array_equal(test_matrix.toarray()[0],
                     test_matrix.toarray()[1]))


# =====================================================
# TASK 3 - TOKENS VS VOCABULARY SIZE
# =====================================================

heading("TASK 3 - TOKENS VS VOCABULARY SIZE")

analyzer = count_vectorizer.build_analyzer()

all_tokens = []

for i, doc in enumerate(documents, start=1):
    tokens = analyzer(doc)
    all_tokens.extend(tokens)
    print(f"D{i} tokens:", tokens)

print("\nTotal token occurrences:", len(all_tokens))
print("Vocabulary size:", len(feature_names))

print("\nWords appearing in more than one document:")

document_frequency = {}

for word in feature_names:
    frequency = sum(
        word in analyzer(doc) for doc in documents
    )
    document_frequency[word] = frequency

for word, frequency in document_frequency.items():
    if frequency > 1:
        print(word, ":", frequency, "documents")


# =====================================================
# TASK 4 - PREPROCESSING INSIDE VECTORIZER
# =====================================================

heading("TASK 4 - PREPROCESSING INSIDE VECTORIZER")

D1 = documents[0]

default_vectorizer = CountVectorizer()
lowercase_false = CountVectorizer(lowercase=False)
stopword_vectorizer = CountVectorizer(stop_words="english")

default_vectorizer.fit([D1])
lowercase_false.fit([D1])
stopword_vectorizer.fit([D1])

print("Default vocabulary:")
print(default_vectorizer.get_feature_names_out())

print("\nLowercase=False vocabulary:")
print(lowercase_false.get_feature_names_out())

print("\nStopwords='english' vocabulary:")
print(stopword_vectorizer.get_feature_names_out())

print("\nStopwords removed:")
print(
    sorted(
        set(default_vectorizer.get_feature_names_out())
        - set(stopword_vectorizer.get_feature_names_out())
    )
)

print("\nChecking 'not' with English stopwords:")

stopword_all = CountVectorizer(stop_words="english")
stopword_all.fit(documents)

print("'not' in vocabulary?",
      "not" in stopword_all.vocabulary_)


# =====================================================
# TASK 5 - LIMITING THE VOCABULARY
# =====================================================

heading("TASK 5 - LIMITING THE VOCABULARY")

min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

print("Full vocabulary size:", len(feature_names))

print("\nVocabulary with min_df=2:")
print(min_df_vectorizer.get_feature_names_out())
print("Size:", len(min_df_vectorizer.get_feature_names_out()))

print("\nVocabulary with max_features=5:")
print(max_features_vectorizer.get_feature_names_out())
print("Size:", len(max_features_vectorizer.get_feature_names_out()))


# =====================================================
# TASK 6 - N-GRAMS
# =====================================================

heading("TASK 6 - N-GRAMS")

D5 = documents[4]

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))

unigram_vectorizer.fit([D5])
bigram_vectorizer.fit([D5])

print("Unigram vocabulary:")
print(unigram_vectorizer.get_feature_names_out())

print("\nUnigram + Bigram vocabulary:")
print(bigram_vectorizer.get_feature_names_out())

print("\nImportant bigram: not good")

print("\nVocabulary growth on D1-D5:")

for n in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=n)
    vectorizer.fit(documents)

    print(
        f"ngram_range={n}:",
        len(vectorizer.get_feature_names_out()),
        "features"
    )


# =====================================================
# TASK 7 - TF-IDF
# =====================================================

heading("TASK 7 - TF-IDF")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_features = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF Matrix (rounded to 2 decimals):")

print(np.round(tfidf_matrix.toarray(), 2))

count_array = count_matrix.toarray()
tfidf_array = tfidf_matrix.toarray()

for doc_index, doc_name in [(0, "D1"), (1, "D2")]:

    print(f"\n{doc_name} - Word Counts and TF-IDF Weights")

    for i, word in enumerate(feature_names):
        count = count_array[doc_index, i]

        if count > 0:
            tfidf_index = list(tfidf_features).index(word)
            weight = tfidf_array[doc_index, tfidf_index]

            print(
                f"{word:12} Count: {count} "
                f"TF-IDF: {weight:.4f}"
            )


# =====================================================
# TASK 8 - TF-IDF BY HAND
# =====================================================

heading("TASK 8 - TF-IDF BY HAND")

N = len(documents)

D2_tokens = analyzer(documents[1])
total_D2_tokens = len(D2_tokens)

print("Total documents:", N)
print("D2 tokens:", D2_tokens)
print("Total D2 tokens:", total_D2_tokens)

for word in ["cash", "today"]:

    count = D2_tokens.count(word)

    df = document_frequency[word]

    tf = count / total_D2_tokens

    idf = np.log(N / df)

    tfidf_hand = tf * idf

    sklearn_index = list(tfidf_features).index(word)
    sklearn_weight = tfidf_array[1, sklearn_index]

    print(f"\nWord: {word}")
    print("Count:", count)
    print("Document Frequency:", df)
    print(f"TF = {count}/{total_D2_tokens} = {tf:.4f}")
    print(f"IDF = log({N}/{df}) = {idf:.4f}")
    print(f"TF-IDF = {tf:.4f} * {idf:.4f} = {tfidf_hand:.4f}")
    print(f"Scikit-learn TF-IDF = {sklearn_weight:.4f}")


# =====================================================
# TASK 9 - SPARSE MATRIX
# =====================================================

heading("TASK 9 - SPARSE MATRIX")

total_cells = count_matrix.shape[0] * count_matrix.shape[1]

nonzero_cells = count_matrix.nnz

zero_cells = total_cells - nonzero_cells

percentage_zeros = (zero_cells / total_cells) * 100

print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print(f"Percentage of zeros: {percentage_zeros:.2f}%")

print("\nSparse matrix storage format:")
print(count_matrix.getformat())


# =====================================================
# TASK 10 - FIT AND TRANSFORM
# =====================================================

heading("TASK 10 - FIT AND TRANSFORM")

training_documents = documents[:4]

training_vectorizer = CountVectorizer()

training_vectorizer.fit(training_documents)

print("Training vocabulary:")
print(training_vectorizer.get_feature_names_out())

new_matrix = training_vectorizer.transform(new_texts)

print("\nNew text vectors:")

for text, vector in zip(new_texts, new_matrix.toarray()):
    print(text)
    print(vector)

training_vocabulary = set(training_vectorizer.get_feature_names_out())

for i, text in enumerate(new_texts, start=1):

    tokens = analyzer(text)

    oov_words = [
        word for word in tokens
        if word not in training_vocabulary
    ]

    print(f"\nN{i} Out-of-vocabulary words:")
    print(oov_words)


# =====================================================
# TASK 11 - DATA LEAKAGE
# =====================================================

heading("TASK 11 - DATA LEAKAGE")

train_only_vectorizer = CountVectorizer()
train_only_vectorizer.fit(training_documents)

combined_documents = training_documents + new_texts

combined_vectorizer = CountVectorizer()
combined_vectorizer.fit(combined_documents)

train_vocab = set(train_only_vectorizer.get_feature_names_out())

combined_vocab = set(combined_vectorizer.get_feature_names_out())

leaked_words = sorted(combined_vocab - train_vocab)

print("Training-only vocabulary:")
print(sorted(train_vocab))

print("\nTraining + Test vocabulary:")
print(sorted(combined_vocab))

print("\nWords leaked from test set:")
print(leaked_words)


# =====================================================
# TASK 12 - NO MEANING / COSINE SIMILARITY
# =====================================================

heading("TASK 12 - COSINE SIMILARITY")

review_vectorizer = CountVectorizer()

review_matrix = review_vectorizer.fit_transform(reviews)

review_features = review_vectorizer.get_feature_names_out()

print("Review vocabulary:")
print(review_features)

print("\nReview vectors:")

for i, vector in enumerate(review_matrix.toarray(), start=1):
    print(f"R{i}:", vector)

similarity_matrix = cosine_similarity(review_matrix)

print("\nCosine Similarity Matrix:")
print(np.round(similarity_matrix, 4))

print("\nPairwise Similarities:")

print("R1 vs R2:", round(similarity_matrix[0, 1], 4))
print("R1 vs R3:", round(similarity_matrix[0, 2], 4))
print("R2 vs R3:", round(similarity_matrix[1, 2], 4))

if similarity_matrix[0, 1] > similarity_matrix[0, 2]:
    print("R1 is more similar to R2.")
elif similarity_matrix[0, 1] < similarity_matrix[0, 2]:
    print("R1 is more similar to R3.")
else:
    print("R1 is equally similar to R2 and R3.")


# =====================================================
# END OF ASSIGNMENT
# =====================================================

heading("ALL 12 PRACTICAL TASKS COMPLETED")

print("Assignment 3 - Text Vectorization")
print("Python script executed successfully.")