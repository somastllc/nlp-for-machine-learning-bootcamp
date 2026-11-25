
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=200)


# Documents: Training corpus

documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

# New texts: Test set

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


# 1) One-hot encoding of single words

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

for word in "win free cash".split():
    one_hot = [
        1 if vocabulary_word == word else 0
        for vocabulary_word in one_hot_vocabulary
    ]
    print(word, "->", one_hot)


# 2) Bag of Words and document-term matrix
# fit learns the vocabulary.
# Each document becomes one row.
# Each vocabulary word becomes one column.


count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print("\n=== 2) BAG OF WORDS ===")
print("Documents:", documents)

print("Vocabulary:",
      count_vectorizer.get_feature_names_out().tolist())

print("Shape (rows, columns):", count_matrix.shape)

print("Document-Term Matrix:")
print(count_matrix.toarray())

# Transform two sentences with the same vocabulary
print("\n'win free phone':",
      count_vectorizer.transform(["win free phone"]).toarray())

print("'phone free win':",
      count_vectorizer.transform(["phone free win"]).toarray())




# 3) Tokens vs vocabulary size
# Tokens count word occurrences.
# Vocabulary size counts unique words.
# Document frequency counts how many documents contain each word.


analyzer = count_vectorizer.build_analyzer()

all_tokens = []
for document in documents:
    all_tokens.extend(analyzer(document))

vocabulary = count_vectorizer.get_feature_names_out()
document_frequency = np.asarray(count_matrix.getnnz(axis=0)).flatten()

print("\n=== 3) TOKENS vs VOCABULARY SIZE ===")
print("Total tokens:", len(all_tokens))
print("Vocabulary size:", len(vocabulary))

print("Words appearing in more than one document:")
for word, df in zip(vocabulary, document_frequency):
    if df > 1:
        print(word, "->", df, "documents")


# 4) Preprocessing inside the vectorizer
# Default: lowercase, punctuation removal and token selection.
# lowercase=False preserves uppercase and lowercase differences.
# English stop words remove common words.


d1 = [documents[0]]

default_vectorizer = CountVectorizer()
no_lowercase_vectorizer = CountVectorizer(lowercase=False)
stopword_vectorizer = CountVectorizer(stop_words="english")

default_vectorizer.fit(d1)
no_lowercase_vectorizer.fit(d1)
stopword_vectorizer.fit(d1)

print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")
print("D1:", documents[0])

print("Default vocabulary:",
      default_vectorizer.get_feature_names_out().tolist())

print("lowercase=False vocabulary:",
      no_lowercase_vectorizer.get_feature_names_out().tolist())

print("stop_words='english' vocabulary:",
      stopword_vectorizer.get_feature_names_out().tolist())

print("\nCheck 'not' in English stop words:")
all_stopword_vectorizer = CountVectorizer(stop_words="english")
all_stopword_vectorizer.fit(documents)

all_stopword_vocabulary = all_stopword_vectorizer.get_feature_names_out().tolist()

print("Vocabulary with English stop words:", all_stopword_vocabulary)
print("Is 'not' in vocabulary?", "not" in all_stopword_vocabulary)


# 5) Limiting the vocabulary
# min_df=2 keeps words appearing in at least 2 documents.
# max_features=5 keeps the 5 most frequent features.


min_df_vectorizer = CountVectorizer(min_df=2)
max_features_vectorizer = CountVectorizer(max_features=5)

min_df_vectorizer.fit(documents)
max_features_vectorizer.fit(documents)

min_df_vocabulary = min_df_vectorizer.get_feature_names_out().tolist()
max_features_vocabulary = max_features_vectorizer.get_feature_names_out().tolist()

print("\n=== 5) LIMITING THE VOCABULARY ===")
print("Full vocabulary:", vocabulary.tolist())
print("Full vocabulary size:", len(vocabulary))

print("\nmin_df=2 vocabulary:", min_df_vocabulary)
print("min_df=2 size:", len(min_df_vocabulary))

print("\nmax_features=5 vocabulary:", max_features_vocabulary)
print("max_features=5 size:", len(max_features_vocabulary))



# 6) N-grams
# Unigrams are single words.
# Bigrams are two neighboring words.
# Trigrams are three neighboring words.


d5 = [documents[4]]

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))

unigram_vectorizer.fit(d5)
bigram_vectorizer.fit(d5)

print("\n=== 6) N-GRAMS ===")
print("D5:", documents[4])

print("Unigrams:",
      unigram_vectorizer.get_feature_names_out().tolist())

print("Unigrams and bigrams:",
      bigram_vectorizer.get_feature_names_out().tolist())

print("\nN-gram vocabulary sizes for D1-D5:")

for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    ngram_vectorizer = CountVectorizer(ngram_range=ngram_range)
    ngram_vectorizer.fit(documents)

    print(
        "ngram_range =", ngram_range,
        "| Vocabulary size =",
        len(ngram_vectorizer.get_feature_names_out())
    )



# 7) TF-IDF
# TF-IDF gives higher importance to words that are more informative.
# The weights are not probabilities.


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out()
count_vocabulary = count_vectorizer.get_feature_names_out()

print("\n=== 7) TF-IDF ===")
print("Vocabulary:", tfidf_vocabulary.tolist())
print("TF-IDF matrix (rounded to 2 decimals):")
print(tfidf_matrix.toarray().round(2))

print("\nCount vs TF-IDF for D1 and D2:")

count_array = count_matrix.toarray()
tfidf_array = tfidf_matrix.toarray()

for document_index in [0, 1]:
    print("\nDocument:", "D" + str(document_index + 1))

    for word, count, weight in zip(
        count_vocabulary,
        count_array[document_index],
        tfidf_array[document_index]
    ):
        if count > 0:
            print(
                word,
                "-> count:", count,
                "| TF-IDF:", round(weight, 3)
            )


# 8) TF-IDF by hand for D2
# TF = count / total words in document
# IDF = log(N / df)
# TF-IDF = TF * IDF


N = len(documents)
d2_counts = count_vectorizer.transform([documents[1]]).toarray()[0]
total_d2_tokens = int(d2_counts.sum())

print("\n=== 8) TF-IDF BY HAND FOR D2 ===")
print("Document:", documents[1])
print("Total documents (N):", N)
print("Total tokens in D2:", total_d2_tokens)

for word in ["cash", "today"]:
    word_index = list(count_vocabulary).index(word)

    count = int(d2_counts[word_index])
    df = int(document_frequency[word_index])

    tf = count / total_d2_tokens
    idf = np.log(N / df)
    tfidf_hand = tf * idf

    print("\nWord:", word)
    print("Count:", count)
    print("Document frequency (df):", df)
    print("TF =", count, "/", total_d2_tokens, "=", round(tf, 6))
    print("IDF = log(", N, "/", df, ") =", round(idf, 6))
    print("TF-IDF =", round(tf, 6), "*", round(idf, 6),
          "=", round(tfidf_hand, 6))


# 9) Sparse matrix
# CountVectorizer stores a sparse matrix.
# Non-zero cells contain word counts.


total_cells = count_matrix.shape[0] * count_matrix.shape[1]
non_zero_cells = count_matrix.nnz
zero_cells = total_cells - non_zero_cells
zero_percentage = (zero_cells / total_cells) * 100

print("\n=== 9) SPARSE MATRIX ===")
print("Matrix shape:", count_matrix.shape)
print("Total cells:", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(zero_percentage, 2), "%")



# 10) Fit on training text, transform new text
# Fit learns vocabulary from D1-D4 only.
# Transform uses that vocabulary on N1 and N2.
# Unknown words are out-of-vocabulary (OOV).


train_texts = documents[:4]

train_vectorizer = CountVectorizer()
X_train = train_vectorizer.fit_transform(train_texts)

new_matrix = train_vectorizer.transform(new_texts)

train_vocabulary = train_vectorizer.get_feature_names_out().tolist()
train_vocabulary_set = set(train_vocabulary)

print("\n=== 10) FIT AND TRANSFORM ===")
print("Training documents: D1-D4")
print("Training vocabulary:", train_vocabulary)

print("\nTraining matrix:")
print(X_train.toarray())

print("\nNew texts and their vectors:")

for text, vector in zip(new_texts, new_matrix.toarray()):
    print(text, "->", vector)

print("\nOut-of-vocabulary words:")

for text in new_texts:
    text_tokens = train_vectorizer.build_analyzer()(text)
    oov_words = sorted(set(text_tokens) - train_vocabulary_set)

    print(text, "->", oov_words)


# 11) Data leakage
# Correct approach: fit on training data only.
# Incorrect approach: fit on training and test data together.


training_vectorizer = CountVectorizer()
training_vectorizer.fit(train_texts)

leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(train_texts + new_texts)

training_vocabulary = set(
    training_vectorizer.get_feature_names_out().tolist()
)

leaky_vocabulary = set(
    leaky_vectorizer.get_feature_names_out().tolist()
)

leaked_words = sorted(leaky_vocabulary - training_vocabulary)

print("\n=== 11) DATA LEAKAGE ===")
print("Training vocabulary:",
      sorted(training_vocabulary))

print("Vocabulary fitted on training + test:",
      sorted(leaky_vocabulary))

print("Words leaked from test set:",
      leaked_words)


# 12) No meaning: review comparison
# CountVectorizer counts words but does not understand their meanings.
# Cosine similarity compares the numerical vectors.


meaning_vectorizer = CountVectorizer()
meaning_matrix = meaning_vectorizer.fit_transform(reviews)

similarity_matrix = cosine_similarity(meaning_matrix)

print("\n=== 12) NO MEANING: REVIEW COMPARISON ===")
print("Reviews:", reviews)

print("Vocabulary:",
      meaning_vectorizer.get_feature_names_out().tolist())

print("\nReview vectors:")
for review, vector in zip(reviews, meaning_matrix.toarray()):
    print(review, "->", vector)

print("\nCosine similarity matrix:")
print(similarity_matrix.round(3))

print("\nPairwise cosine similarities:")
print("R1 vs R2:", round(similarity_matrix[0, 1], 3))
print("R1 vs R3:", round(similarity_matrix[0, 2], 3))
print("R2 vs R3:", round(similarity_matrix[1, 2], 3))
