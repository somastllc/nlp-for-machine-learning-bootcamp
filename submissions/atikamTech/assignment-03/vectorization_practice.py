
import re
import math
import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

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

D1, D2, D3, D4, D5 = documents
N1, N2 = new_texts
R1, R2, R3 = reviews


def heading(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_vocabulary(vectorizer):
    print(list(vectorizer.get_feature_names_out()))


# ------------------------------------------------------------
# TASK 1: ONE-HOT ENCODING
# ------------------------------------------------------------

heading("TASK 1: ONE-HOT ENCODING")

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]
word_to_index = {word: index for index, word in enumerate(vocabulary)}

def one_hot(word):
    vector = [0] * len(vocabulary)
    if word in word_to_index:
        vector[word_to_index[word]] = 1
    return vector

print("Vocabulary:", vocabulary)

for word in ["free", "cash"]:
    print(f"{word}: {one_hot(word)}")

sentence_words = ["win", "free", "cash"]
print("\nSentence:", " ".join(sentence_words))

for word in sentence_words:
    print(f"{word}: {one_hot(word)}")

print("Number of vectors:", len(sentence_words))
print("Length of each vector:", len(vocabulary))


# ------------------------------------------------------------
# TASK 2: BAG OF WORDS
# ------------------------------------------------------------

heading("TASK 2: BAG OF WORDS")

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

features = count_vectorizer.get_feature_names_out()

print("Vocabulary:")
show_vocabulary(count_vectorizer)

print("\nMatrix shape:", count_matrix.shape)
print("\nFull document-term matrix:")
print(count_matrix.toarray())

print("\nD1 word counts:")
d1_counts = count_matrix.toarray()[0]
for word, count in zip(features, d1_counts):
    if count > 0:
        print(f"{word}: {count}")

print("\nTransforming two sentences with the same words:")
pair_texts = ["win free phone", "phone free win"]
pair_matrix = count_vectorizer.transform(pair_texts)

print(pair_texts[0], "->", pair_matrix.toarray()[0])
print(pair_texts[1], "->", pair_matrix.toarray()[1])
print("Are the vectors equal?", np.array_equal(
    pair_matrix.toarray()[0], pair_matrix.toarray()[1]
))


# ------------------------------------------------------------
# TASK 3: TOKENS VS VOCABULARY SIZE
# ------------------------------------------------------------

heading("TASK 3: TOKENS VS VOCABULARY SIZE")

analyzer = count_vectorizer.build_analyzer()

all_tokens = []
document_frequency = Counter()

for document in documents:
    tokens = analyzer(document)
    all_tokens.extend(tokens)
    document_frequency.update(set(tokens))

print("Tokens in each document:")
for index, document in enumerate(documents, start=1):
    tokens = analyzer(document)
    print(f"D{index}: {tokens} ({len(tokens)} tokens)")

print("\nTotal token occurrences:", len(all_tokens))
print("Vocabulary size:", len(count_vectorizer.vocabulary_))

repeated_words = sorted(
    word for word, frequency in document_frequency.items()
    if frequency > 1
)

print("Words appearing in more than one document:")
print(repeated_words)


# ------------------------------------------------------------
# TASK 4: PREPROCESSING INSIDE THE VECTORIZER
# ------------------------------------------------------------

heading("TASK 4: PREPROCESSING INSIDE THE VECTORIZER")

default_vectorizer = CountVectorizer()
default_vectorizer.fit([D1])

case_sensitive_vectorizer = CountVectorizer(lowercase=False)
case_sensitive_vectorizer.fit([D1])

stopword_vectorizer = CountVectorizer(stop_words="english")
stopword_vectorizer.fit([D1])

print("Default vocabulary:")
show_vocabulary(default_vectorizer)

print("\nVocabulary with lowercase=False:")
show_vocabulary(case_sensitive_vectorizer)

print("\nVocabulary with stop_words='english':")
show_vocabulary(stopword_vectorizer)

print("\nD1 tokens under default settings:")
print(default_vectorizer.build_analyzer()(D1))

print("\nD1 tokens with lowercase=False:")
print(case_sensitive_vectorizer.build_analyzer()(D1))

print("\nD1 tokens with English stopwords:")
print(stopword_vectorizer.build_analyzer()(D1))

stopword_all_vectorizer = CountVectorizer(stop_words="english")
stopword_all_matrix = stopword_all_vectorizer.fit_transform(documents)

print("\nVocabulary with English stopwords on D1-D5:")
show_vocabulary(stopword_all_vectorizer)

print("Is 'not' in the vocabulary?",
      "not" in stopword_all_vectorizer.vocabulary_)


# ------------------------------------------------------------
# TASK 5: LIMITING THE VOCABULARY
# ------------------------------------------------------------

heading("TASK 5: LIMITING THE VOCABULARY")

min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

print("Full vocabulary size:", len(count_vectorizer.vocabulary_))

print("\nVocabulary with min_df=2:")
show_vocabulary(min_df_vectorizer)
print("Size:", len(min_df_vectorizer.vocabulary_))

print("\nVocabulary with max_features=5:")
show_vocabulary(max_features_vectorizer)
print("Size:", len(max_features_vectorizer.vocabulary_))


# ------------------------------------------------------------
# TASK 6: N-GRAMS
# ------------------------------------------------------------

heading("TASK 6: N-GRAMS")

unigram_D5 = CountVectorizer(ngram_range=(1, 1))
bigram_D5 = CountVectorizer(ngram_range=(1, 2))

unigram_D5.fit([D5])
bigram_D5.fit([D5])

print("D5 unigram vocabulary:")
show_vocabulary(unigram_D5)

print("\nD5 unigram + bigram vocabulary:")
show_vocabulary(bigram_D5)

print("\nN-gram vocabulary sizes on D1-D5:")

for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(documents)
    print(f"{ngram_range}: {len(vectorizer.vocabulary_)} features")


# ------------------------------------------------------------
# TASK 7: TF-IDF
# ------------------------------------------------------------

heading("TASK 7: TF-IDF")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_features = tfidf_vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()
count_array = count_matrix.toarray()

print("TF-IDF vocabulary:")
print(list(tfidf_features))

print("\nTF-IDF matrix rounded to 2 decimals:")
print(np.round(tfidf_array, 2))

for document_index, document_name in [(0, "D1"), (1, "D2")]:
    print(f"\n{document_name}: word, count, TF-IDF weight")
    for word_index, word in enumerate(tfidf_features):
        count = count_array[document_index, word_index]
        weight = tfidf_array[document_index, word_index]
        if count > 0:
            print(f"{word:15} count={count:<3} TF-IDF={weight:.4f}")


# ------------------------------------------------------------
# TASK 8: TF-IDF BY HAND
# ------------------------------------------------------------

heading("TASK 8: TF-IDF BY HAND")

N = len(documents)
d2_tokens = count_vectorizer.build_analyzer()(D2)
d2_token_count = len(d2_tokens)

print("Number of documents (N):", N)
print("D2 analyzed tokens:", d2_tokens)
print("D2 token count:", d2_token_count)

d2_row = count_matrix.toarray()[1]

for word in ["cash", "today"]:
    word_index = count_vectorizer.vocabulary_[word]
    term_count = d2_row[word_index]
    df = int(np.count_nonzero(count_array[:, word_index]))

    tf = term_count / d2_token_count
    idf = math.log(N / df)
    textbook_tfidf = tf * idf

    sklearn_weight = tfidf_array[1, tfidf_vectorizer.vocabulary_[word]]

    print(f"\nWord: {word}")
    print("Count in D2:", term_count)
    print("Document frequency:", df)
    print(f"TF = count / total tokens = {term_count}/{d2_token_count} = {tf:.6f}")
    print(f"IDF = log(N / df) = log({N}/{df}) = {idf:.6f}")
    print(f"Textbook TF-IDF = TF * IDF = {textbook_tfidf:.6f}")
    print(f"scikit-learn normalized TF-IDF = {sklearn_weight:.6f}")

print("\nTextbook ranking:")
print("cash has a higher textbook TF-IDF weight than today.")


# ------------------------------------------------------------
# TASK 9: SPARSE MATRIX
# ------------------------------------------------------------

heading("TASK 9: SPARSE MATRIX")

rows, columns = count_matrix.shape
total_cells = rows * columns
nonzero_cells = count_matrix.nnz
zero_cells = total_cells - nonzero_cells
zero_percentage = (zero_cells / total_cells) * 100

print("Matrix shape:", count_matrix.shape)
print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print(f"Percentage of zeros: {zero_percentage:.2f}%")


# ------------------------------------------------------------
# TASK 10: FIT AND TRANSFORM
# ------------------------------------------------------------

heading("TASK 10: FIT AND TRANSFORM")

training_documents = documents[:4]

training_vectorizer = CountVectorizer()
training_matrix = training_vectorizer.fit_transform(training_documents)

print("Training vocabulary:")
show_vocabulary(training_vectorizer)

new_matrix = training_vectorizer.transform(new_texts)

print("\nNew text vectors:")
for text, vector in zip(new_texts, new_matrix.toarray()):
    print(text)
    print(vector)

training_vocabulary = set(training_vectorizer.vocabulary_)

print("\nOut-of-vocabulary words:")
for text_name, text in zip(["N1", "N2"], new_texts):
    tokens = training_vectorizer.build_analyzer()(text)
    oov_words = sorted(set(tokens) - training_vocabulary)
    print(f"{text_name}: {oov_words}")


# ------------------------------------------------------------
# TASK 11: DATA LEAKAGE
# ------------------------------------------------------------

heading("TASK 11: DATA LEAKAGE")

train_only_vectorizer = CountVectorizer()
train_only_vectorizer.fit(training_documents)

combined_vectorizer = CountVectorizer()
combined_vectorizer.fit(training_documents + new_texts)

train_vocabulary = set(train_only_vectorizer.vocabulary_)
combined_vocabulary = set(combined_vectorizer.vocabulary_)

leaked_words = sorted(combined_vocabulary - train_vocabulary)

print("Training-only vocabulary:")
print(sorted(train_vocabulary))

print("\nVocabulary after fitting on training + test texts:")
print(sorted(combined_vocabulary))

print("\nWords introduced by the test set:")
print(leaked_words)


# ------------------------------------------------------------
# TASK 12: VECTORS DO NOT CAPTURE MEANING
# ------------------------------------------------------------

heading("TASK 12: VECTORS DO NOT CAPTURE MEANING")

review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(reviews)

print("Review vocabulary:")
show_vocabulary(review_vectorizer)

print("\nReview vectors:")
for review, vector in zip(reviews, review_matrix.toarray()):
    print(review)
    print(vector)

similarity_matrix = cosine_similarity(review_matrix)

print("\nCosine similarity matrix:")
print(np.round(similarity_matrix, 4))

print("\nPairwise similarities:")
print("R1 vs R2:", round(similarity_matrix[0, 1], 4))
print("R1 vs R3:", round(similarity_matrix[0, 2], 4))
print("R2 vs R3:", round(similarity_matrix[1, 2], 4))

print("\nEND OF ASSIGNMENT 3 PRACTICAL")
