# Assignment Three - Vectorization - Practical
# Part B - Practical


import math
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=200)

# --------------------------------
# SETUP: DOCUMENTS
# Training corpus, new texts and reviews.
# --------------------------------

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
# DISCUSSION: UNDERSTANDING THE DATA
# D1-D5 are training documents.
# N1-N2 are new texts for testing.
# R1-R3 are reviews for comparing word representations.
# --------------------------------

print("\n=== DISCUSSION: THE DOCUMENTS ===")
print("Training documents:", len(documents))
print("New texts:", len(new_texts))
print("Reviews:", len(reviews))

for i, document in enumerate(documents, start=1):
    print(f"D{i}: {document}")

# --------------------------------
# 1) One-hot encoding of single words
# Each vocabulary word gets a vector with one 1.
# The number of vectors depends on the number of words.
# --------------------------------

one_hot_vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

print("\n=== 1) ONE-HOT ENCODING ===")
print("Vocabulary:", one_hot_vocabulary)

for word in "free cash".split():
    one_hot = [
        1 if vocabulary_word == word else 0
        for vocabulary_word in one_hot_vocabulary
    ]
    print(word, "->", one_hot)

print("\nOne-hot vectors for 'win free cash':")

sentence = "win free cash".split()

for word in sentence:
    one_hot = [
        1 if vocabulary_word == word else 0
        for vocabulary_word in one_hot_vocabulary
    ]
    print(word, "->", one_hot)

print("Number of vectors:", len(sentence))
print("Features per vector:", len(one_hot_vocabulary))

# Discussion:
# Each word gets its own vector.
# A sentence with three words produces three vectors.
# One-hot encoding does not directly represent a whole document.


# --------------------------------
# 2) Bag of Words and document-term matrix
# fit learns the vocabulary.
# Rows represent documents; columns represent words.
# Cells contain word counts.
# --------------------------------

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print("\n=== 2) BAG OF WORDS ===")
print("Vocabulary:", count_vectorizer.get_feature_names_out().tolist())
print("Shape (rows, columns):", count_matrix.shape)
print("Document-term matrix:")
print(count_matrix.toarray())

# Find words that appear twice in D1.
d1_counts = count_matrix.toarray()[0]
vocabulary = count_vectorizer.get_feature_names_out()

print("\nWords with count 2 in D1:")

for word, count in zip(vocabulary, d1_counts):
    if count == 2:
        print(word, "->", count)

# Compare different word orders.
comparison = ["win free phone", "phone free win"]

print("\n'win free phone':", count_vectorizer.transform([comparison[0]]).toarray())
print("'phone free win':", count_vectorizer.transform([comparison[1]]).toarray())

print(
    "Are the vectors the same:",
    np.array_equal(
        count_vectorizer.transform([comparison[0]]).toarray(),
        count_vectorizer.transform([comparison[1]]).toarray()
    )
)

# Discussion:
# Bag of Words counts words but does not preserve their order.
# This can work for some topic-classification tasks.
# It can cause problems when word order changes the meaning.


# --------------------------------
# 3) Tokens vs vocabulary size
# Tokens count every word occurrence.
# Vocabulary size counts unique words.
# --------------------------------

analyzer = count_vectorizer.build_analyzer()
token_lists = [analyzer(document) for document in documents]

all_tokens = [
    token
    for document_tokens in token_lists
    for token in document_tokens
]

print("\n=== 3) TOKENS VS VOCABULARY SIZE ===")

for i, tokens in enumerate(token_lists, start=1):
    print(f"D{i} tokens:", tokens)
    print(f"D{i} token count:", len(tokens))

print("Total tokens:", len(all_tokens))
print("Vocabulary size:", len(vocabulary))

# Document frequency: number of documents containing each word.
document_frequency = {
    word: sum(word in tokens for tokens in token_lists)
    for word in vocabulary
}

repeated_words = {
    word: df
    for word, df in document_frequency.items()
    if df > 1
}

print("\nWords appearing in more than one document:")

for word, df in sorted(repeated_words.items()):
    print(word, "->", df, "documents")

# Discussion:
# Tokens count repeated words.
# Vocabulary size counts each unique word once.
# Document-term matrices become sparse because most documents
# contain only a small part of the full vocabulary.


# --------------------------------
# 4) Preprocessing inside the vectorizer
# Default: lowercase, punctuation removal and tokenization.
# lowercase=False keeps uppercase and lowercase forms separate.
# stop_words="english" removes selected common English words.
# --------------------------------

print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")

d1 = [documents[0]]

default_vectorizer = CountVectorizer()
case_vectorizer = CountVectorizer(lowercase=False)
stopword_vectorizer = CountVectorizer(stop_words="english")

print("D1:", documents[0])

print(
    "Default:",
    default_vectorizer.fit(d1).get_feature_names_out().tolist()
)

print(
    "lowercase=False:",
    case_vectorizer.fit(d1).get_feature_names_out().tolist()
)

print(
    "stop_words='english':",
    stopword_vectorizer.fit(d1).get_feature_names_out().tolist()
)

print("\nDefault analyzer tokens:")
print(default_vectorizer.build_analyzer()(documents[0]))

print("\nTokens with lowercase=False:")
print(case_vectorizer.build_analyzer()(documents[0]))

# Find the words removed by the English stopword list.
default_words = set(default_vectorizer.get_feature_names_out())
stopword_words = set(stopword_vectorizer.get_feature_names_out())

print("\nWords removed by the English stopword list:")
print(sorted(default_words - stopword_words))

# Check whether "not" is removed from the full corpus.
all_stopword_vectorizer = CountVectorizer(stop_words="english")
all_stopword_vectorizer.fit(documents)

print("\nIs 'not' in the vocabulary?")
print("not" in all_stopword_vectorizer.get_feature_names_out())

print("\nD5 after English stopword removal:")
print(all_stopword_vectorizer.build_analyzer()(documents[4]))

# Discussion:
# The default vectorizer lowercases words, so FREE and free become free.
# The default token pattern ignores punctuation and one-letter words.
# English stopwords can remove words such as "not".
# Removing "not" can change the meaning of D5.


# --------------------------------
# 5) Limiting the vocabulary
# min_df=2 keeps words appearing in at least two documents.
# max_features=5 limits the vocabulary to five selected features.
# --------------------------------

print("\n=== 5) LIMITING THE VOCABULARY ===")

full_vocabulary = CountVectorizer().fit(documents).get_feature_names_out()

min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

print("Full vocabulary size:", len(full_vocabulary))
print("Full vocabulary:", full_vocabulary.tolist())

print("\nmin_df=2:")
print(min_df_vectorizer.get_feature_names_out().tolist())
print("Vocabulary size:", len(min_df_vectorizer.get_feature_names_out()))

print("\nmax_features=5:")
print(max_features_vectorizer.get_feature_names_out().tolist())
print("Vocabulary size:", len(max_features_vectorizer.get_feature_names_out()))

# Discussion:
# min_df removes words that appear in fewer than two documents.
# max_features selects a limited number of frequent features.
# These settings can reduce the size of the vocabulary.


# --------------------------------
# 6) N-grams
# Unigrams contain one word.
# Bigrams contain two neighboring words.
# Trigrams contain three neighboring words.
# --------------------------------

sentence = [documents[4]]

print("\n=== 6) N-GRAMS ===")
print("D5:", sentence[0])

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))

print(
    "Unigrams:",
    unigram_vectorizer.fit(sentence).get_feature_names_out().tolist()
)

print(
    "Unigrams and bigrams:",
    bigram_vectorizer.fit(sentence).get_feature_names_out().tolist()
)

print("\nImportant bigrams:")

d5_bigrams = set(bigram_vectorizer.get_feature_names_out())

for bigram in ["not good", "were great"]:
    print(bigram, "->", bigram in d5_bigrams)

print("\nVocabulary sizes on D1-D5:")

for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(documents)

    print(
        ngram_range,
        "->",
        len(vectorizer.get_feature_names_out()),
        "features"
    )

# Discussion:
# Bigrams preserve pairs of neighboring words.
# "not good" keeps a negative expression together.
# Larger n-grams create more features and require more memory.


# --------------------------------
# 7) TF-IDF
# TF-IDF gives words weights based on their document importance.
# Words found in fewer documents generally receive higher IDF.
# --------------------------------

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()
count_array = count_matrix.toarray()

print("\n=== 7) TF-IDF ===")
print("Vocabulary:", tfidf_vocabulary.tolist())
print("TF-IDF matrix:")
print(tfidf_array.round(2))

print("\nCount vs TF-IDF for D1:")

for word in tfidf_vocabulary:
    count = count_array[0, np.where(vocabulary == word)[0][0]]
    weight = tfidf_array[0, np.where(tfidf_vocabulary == word)[0][0]]

    if count > 0:
        print(word, "-> count:", count, "| TF-IDF:", round(weight, 3))

print("\nCount vs TF-IDF for D2:")

for word in tfidf_vocabulary:
    count = count_array[1, np.where(vocabulary == word)[0][0]]
    weight = tfidf_array[1, np.where(tfidf_vocabulary == word)[0][0]]

    if count > 0:
        print(word, "-> count:", count, "| TF-IDF:", round(weight, 3))

# Discussion:
# TF-IDF combines term frequency and inverse document frequency.
# Rare words can receive higher weights.
# Scikit-learn applies smoothing and L2 normalization by default.
# TF-IDF values are weights, not probabilities.


# --------------------------------
# 8) TF-IDF by hand using the textbook formula
# TF = count / words in document.
# IDF = log(N / df).
# TF-IDF = TF x IDF.
# --------------------------------

print("\n=== 8) TF-IDF BY HAND FOR D2 ===")

N = len(documents)
d2_tokens = analyzer(documents[1])
words_in_d2 = len(d2_tokens)

print("Number of documents:", N)
print("D2 tokens:", d2_tokens)
print("Words in D2:", words_in_d2)

manual_tfidf = {}

for word in ["cash", "today"]:
    count = d2_tokens.count(word)
    df = document_frequency[word]

    tf = count / words_in_d2
    idf = np.log(N / df)
    tfidf = tf * idf

    manual_tfidf[word] = tfidf

    print("\nWord:", word)
    print("Count:", count)
    print("TF:", round(tf, 4))
    print("Document frequency:", df)
    print("IDF:", round(idf, 4))
    print("TF-IDF:", round(tfidf, 4))

print("\nComparison with scikit-learn:")

for word in ["cash", "today"]:
    index = np.where(tfidf_vocabulary == word)[0][0]

    print(
        word,
        "| textbook:",
        round(manual_tfidf[word], 4),
        "| scikit-learn:",
        round(tfidf_array[1, index], 4)
    )

# Discussion:
# The textbook formula uses log(N / df).
# Scikit-learn uses a smoothed IDF formula and normalizes the vectors.
# The numerical weights differ, but rarer words receive higher IDF.


# --------------------------------
# 9) Sparse matrices
# Sparse matrices contain many zero values.
# They store non-zero values and their positions efficiently.
# --------------------------------

total_cells = count_matrix.shape[0] * count_matrix.shape[1]
non_zero_cells = count_matrix.nnz
zero_cells = total_cells - non_zero_cells

print("\n=== 9) SPARSE MATRIX ===")
print("Matrix shape:", count_matrix.shape)
print("Total cells:", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(zero_cells / total_cells * 100, 2), "%")

# Discussion:
# Most documents do not contain every vocabulary word.
# Sparse storage avoids storing all the zero values.


# --------------------------------
# 10) Fit on training text, transform new text
# fit learns the training vocabulary.
# transform uses that vocabulary on new documents.
# Unknown words are ignored.
# --------------------------------

print("\n=== 10) FIT ON TRAIN, TRANSFORM NEW TEXT ===")

train_texts = documents[:4]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)

train_vocabulary = set(vectorizer.get_feature_names_out())

print("Training vocabulary:")
print(vectorizer.get_feature_names_out().tolist())

print("\nTraining matrix:")
print(X_train.toarray())

X_new = vectorizer.transform(new_texts)

print("\nNew text vectors:")

for i, row in enumerate(X_new.toarray(), start=1):
    print(f"N{i}:", row)

# Find out-of-vocabulary words.
new_analyzer = vectorizer.build_analyzer()

for i, text in enumerate(new_texts, start=1):
    tokens = new_analyzer(text)
    oov_words = sorted(set(tokens) - train_vocabulary)

    print(f"\nN{i}:", text)
    print("Tokens:", tokens)
    print("Out-of-vocabulary words:", oov_words)

# Discussion:
# fit learns the vocabulary.
# transform converts new documents using the learned vocabulary.
# "prize" is unknown because training contains "prizes".
# Stemming or lemmatization can help handle related word forms.


# --------------------------------
# 11) Data leakage
# Correct: fit on training documents only.
# Incorrect: fit on training and test documents together.
# --------------------------------

print("\n=== 11) DATA LEAKAGE ===")

training_vectorizer = CountVectorizer()
training_vectorizer.fit(train_texts)

leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(train_texts + new_texts)

training_vocabulary = set(
    training_vectorizer.get_feature_names_out()
)

leaky_vocabulary = set(
    leaky_vectorizer.get_feature_names_out()
)

leaked_words = sorted(leaky_vocabulary - training_vocabulary)

print("Fitted on training text only:")
print(sorted(training_vocabulary))

print("\nFitted on training and test text:")
print(sorted(leaky_vocabulary))

print("\nWords introduced by the test set:")
print(leaked_words)

# Discussion:
# Data leakage happens when test information enters model preparation.
# The correct order is to fit on training data and transform test data.
# This helps keep evaluation fair.


# --------------------------------
# 12) The main limitation: no meaning
# CountVectorizer treats different words as separate features.
# Cosine similarity compares vectors, not word meanings.
# --------------------------------

meaning_vectorizer = CountVectorizer()
meaning_matrix = meaning_vectorizer.fit_transform(reviews)

similarity_matrix = cosine_similarity(meaning_matrix)

print("\n=== 12) NO MEANING: REVIEW COMPARISON ===")

print("Reviews:", reviews)
print("\nVocabulary:")
print(meaning_vectorizer.get_feature_names_out().tolist())

print("\nReview vectors:")

for i, row in enumerate(meaning_matrix.toarray(), start=1):
    print(f"R{i}:", row)

print("\nCosine similarity matrix:")
print(similarity_matrix.round(4))

print("\nPairwise comparisons:")
print("R1 vs R2:", round(similarity_matrix[0, 1], 4))
print("R1 vs R3:", round(similarity_matrix[0, 2], 4))
print("R2 vs R3:", round(similarity_matrix[1, 2], 4))

print("\nIs R1 more similar to R2 than R3?")

print(similarity_matrix[0, 1] > similarity_matrix[0, 2])

# Discussion:
# "Tasty", "delicious" and "terrible" are separate vocabulary features.
# Bag of Words does not understand that tasty and delicious
# have similar meanings.
# All three reviews share the words "the", "food" and "was".
# Therefore, their count vectors have equal pairwise similarity.


# --------------------------------
# FINAL SUMMARY
# --------------------------------

print("\n=== FINAL SUMMARY ===")

print("1. One-hot encoding represents individual words with binary vectors.")
print("2. Bag of Words represents documents using word counts.")
print("3. Tokens count occurrences; vocabulary size counts unique words.")
print("4. Preprocessing changes which words enter the vocabulary.")
print("5. min_df and max_features can reduce vocabulary size.")
print("6. N-grams preserve some neighboring word order.")
print("7. TF-IDF assigns weights to words based on their importance.")
print("8. Textbook TF-IDF and scikit-learn TF-IDF use different formulas.")
print("9. Sparse matrices save memory by storing non-zero values.")
print("10. fit learns a vocabulary; transform uses it on new text.")
print("11. Data leakage can make model evaluation unfair.")
print("12. Bag of Words counts words but does not understand their meaning.")
