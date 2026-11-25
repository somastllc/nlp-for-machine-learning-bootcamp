# Assignment 3 - Text Vectorization
# Simple version based on Lesson 3 teaching script

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

np.set_printoptions(linewidth=200)


# ============================================================
# DATA
# ============================================================

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


# ============================================================
# 1) ONE-HOT ENCODING
# ============================================================

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

print("\n========================================")
print("TASK 1 - ONE-HOT ENCODING")
print("========================================")

print("Vocabulary:", vocabulary)

print("\nSentence: free cash")

for word in "free cash".split():
    vector = [1 if v == word else 0 for v in vocabulary]
    print(word, "->", vector)

print("\nSentence: win free cash")

for word in "win free cash".split():
    vector = [1 if v == word else 0 for v in vocabulary]
    print(word, "->", vector)

print("\nNumber of word vectors:", len("win free cash".split()))


# ============================================================
# 2) BAG OF WORDS / DOCUMENT-TERM MATRIX
# ============================================================

vectorizer = CountVectorizer()

count_matrix = vectorizer.fit_transform(documents)

print("\n========================================")
print("TASK 2 - BAG OF WORDS")
print("========================================")

print("Vocabulary:")
print(vectorizer.get_feature_names_out().tolist())

print("\nShape:")
print(count_matrix.shape)

print("\nDocument-Term Matrix:")
print(count_matrix.toarray())

print("\nD1 cells with value 2:")

words = vectorizer.get_feature_names_out()
d1 = count_matrix.toarray()[0]

for word, count in zip(words, d1):
    if count == 2:
        print(word, "->", count)

print("\n'win free phone':")
print(vectorizer.transform(["win free phone"]).toarray())

print("\n'phone free win':")
print(vectorizer.transform(["phone free win"]).toarray())


# ============================================================
# 3) TOKENS VS VOCABULARY
# ============================================================

print("\n========================================")
print("TASK 3 - TOKENS VS VOCABULARY")
print("========================================")

# Create vocabulary from D1-D5
task3_vectorizer = CountVectorizer()

task3_matrix = task3_vectorizer.fit_transform(documents)

task3_vocabulary = task3_vectorizer.get_feature_names_out()

# Combine all documents
all_tokens = " ".join(documents).lower()

# Remove punctuation
all_tokens = all_tokens.replace(".", "")
all_tokens = all_tokens.replace(",", "")
all_tokens = all_tokens.replace("!", "")
all_tokens = all_tokens.replace("?", "")

# Split into words
all_tokens = all_tokens.split()

# CountVectorizer ignores one-character words
all_tokens = [word for word in all_tokens if len(word) >= 2]

print("Total tokens:", len(all_tokens))
print("Vocabulary size:", len(task3_vocabulary))

print("\nWords appearing in more than one document:")

for word in task3_vocabulary:

    count = 0

    for document in documents:

        words = document.lower()

        words = words.replace(".", "")
        words = words.replace(",", "")
        words = words.replace("!", "")
        words = words.replace("?", "")

        words = words.split()

        if word in words:
            count = count + 1

    if count > 1:
        print(word, "->", count, "documents")

# ============================================================
# 4) PREPROCESSING
# ============================================================

message = [
    "FREE Prize! Call the office NOW"
]

print("\n========================================")
print("TASK 4 - PREPROCESSING")
print("========================================")

print("Message:")
print(message[0])

print("\nDefault:")
print(
    CountVectorizer()
    .fit(message)
    .get_feature_names_out()
    .tolist()
)

print("\nlowercase=False:")
print(
    CountVectorizer(lowercase=False)
    .fit(message)
    .get_feature_names_out()
    .tolist()
)

print("\nstop_words='english':")
print(
    CountVectorizer(stop_words="english")
    .fit(message)
    .get_feature_names_out()
    .tolist()
)


# Stopwords on all documents

stop_vectorizer = CountVectorizer(stop_words="english")

stop_matrix = stop_vectorizer.fit_transform(documents)

print("\nVocabulary after English stopwords:")
print(stop_vectorizer.get_feature_names_out().tolist())

print("\nIs 'not' in vocabulary?")
print("not" in stop_vectorizer.get_feature_names_out())


# ============================================================
# 5) LIMITING THE VOCABULARY
# ============================================================

print("\n========================================")
print("TASK 5 - LIMITING VOCABULARY")
print("========================================")

full_vectorizer = CountVectorizer()

min_df_vectorizer = CountVectorizer(min_df=2)

max_features_vectorizer = CountVectorizer(max_features=5)

max_df_vectorizer = CountVectorizer(max_df=0.4)

full_vectorizer.fit(documents)
min_df_vectorizer.fit(documents)
max_features_vectorizer.fit(documents)
max_df_vectorizer.fit(documents)

print("Full vocabulary:")
print(full_vectorizer.get_feature_names_out().tolist())

print("\nFull vocabulary size:")
print(len(full_vectorizer.get_feature_names_out()))

print("\nmin_df=2:")
print(min_df_vectorizer.get_feature_names_out().tolist())

print("\nmin_df=2 size:")
print(len(min_df_vectorizer.get_feature_names_out()))

print("\nmax_features=5:")
print(max_features_vectorizer.get_feature_names_out().tolist())

print("\nmax_features=5 size:")
print(len(max_features_vectorizer.get_feature_names_out()))

print("\nmax_df=0.4:")
print(max_df_vectorizer.get_feature_names_out().tolist())


# ============================================================
# 6) N-GRAMS
# ============================================================

print("\n========================================")
print("TASK 6 - N-GRAMS")
print("========================================")

sentence = [
    "The lecture was not good, but the notes were great."
]

unigram_vectorizer = CountVectorizer(
    ngram_range=(1, 1)
)

bigram_vectorizer = CountVectorizer(
    ngram_range=(1, 2)
)

unigram_vectorizer.fit(sentence)
bigram_vectorizer.fit(sentence)

print("Unigrams:")
print(
    unigram_vectorizer
    .get_feature_names_out()
    .tolist()
)

print("\nUnigrams + Bigrams:")
print(
    bigram_vectorizer
    .get_feature_names_out()
    .tolist()
)

print("\nImportant bigram:")
print("not good")


print("\nVocabulary growth on D1-D5:")

for ngram_range in [(1, 1), (1, 2), (1, 3)]:

    v = CountVectorizer(
        ngram_range=ngram_range
    )

    v.fit(documents)

    print(
        ngram_range,
        "->",
        len(v.get_feature_names_out()),
        "features"
    )


# ============================================================
# 7) TF-IDF
# ============================================================

print("\n========================================")
print("TASK 7 - TF-IDF")
print("========================================")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("Vocabulary:")
print(
    tfidf_vectorizer
    .get_feature_names_out()
    .tolist()
)

print("\nTF-IDF Matrix:")
print(
    tfidf_matrix
    .toarray()
    .round(2)
)


# D2 counts and TF-IDF

count_vectorizer = CountVectorizer()

count_vectorizer.fit(documents)

count_matrix = count_vectorizer.transform(documents)

words = count_vectorizer.get_feature_names_out()

d2_counts = count_matrix.toarray()[1]

d2_weights = tfidf_matrix.toarray()[1]

print("\nD2 word | count | TF-IDF")

for word, count, weight in zip(
    words,
    d2_counts,
    d2_weights
):

    if count > 0:

        print(
            word,
            "|",
            count,
            "|",
            round(weight, 4)
        )




# ============================================================
# ============================================================
# 8) TF-IDF BY HAND
# ============================================================

print("\n========================================")
print("TASK 8 - TF-IDF BY HAND")
print("========================================")

N = len(documents)

# D2
d2_words = documents[1].lower()

# Remove punctuation
d2_words = d2_words.replace(".", "")
d2_words = d2_words.replace(",", "")
d2_words = d2_words.replace("!", "")
d2_words = d2_words.replace("?", "")
d2_words = d2_words.split()

number_of_words = len(d2_words)

print("Number of documents:", N)
print("Number of words in D2:", number_of_words)

for word in ["cash", "today"]:

    count = d2_words.count(word)

    df = 0

    for document in documents:

        words = document.lower()

        words = words.replace(".", "")
        words = words.replace(",", "")
        words = words.replace("!", "")
        words = words.replace("?", "")
        words = words.split()

        if word in words:
            df = df + 1

    tf = count / number_of_words

    idf = np.log(N / df)

    tfidf = tf * idf

    print("\nWord:", word)
    print("Count:", count)
    print("Document frequency:", df)
    print("TF:", round(tf, 6))
    print("IDF:", round(idf, 6))
    print("TF-IDF:", round(tfidf, 6))


# ============================================================
# 9) SPARSE MATRIX
# ============================================================

print("\n========================================")
print("TASK 9 - SPARSE MATRIX")
print("========================================")

rows = count_matrix.shape[0]

columns = count_matrix.shape[1]

total_cells = rows * columns

non_zero = count_matrix.nnz

zero = total_cells - non_zero

percentage_zero = (zero / total_cells) * 100

print("Rows:", rows)

print("Columns:", columns)

print("Total cells:", total_cells)

print("Non-zero cells:", non_zero)

print("Zero cells:", zero)

print(
    "Percentage of zeros:",
    round(percentage_zero, 2),
    "%"
)


# ============================================================
# 10) FIT ON TRAINING / TRANSFORM NEW TEXT
# ============================================================

print("\n========================================")
print("TASK 10 - FIT ON TRAIN / TRANSFORM NEW")
print("========================================")

train_documents = documents[:4]

train_vectorizer = CountVectorizer()

X_train = train_vectorizer.fit_transform(
    train_documents
)

print("Training vocabulary:")

print(
    train_vectorizer
    .get_feature_names_out()
    .tolist()
)

print("\nTraining matrix:")

print(X_train.toarray())


for text in new_texts:

    vector = train_vectorizer.transform(
        [text]
    )

    print("\nText:")
    print(text)

    print("Vector:")
    print(vector.toarray())

    words_in_text = text.lower().split()

    vocabulary_words = (
        train_vectorizer
        .get_feature_names_out()
        .tolist()
    )

    oov = []

    for word in words_in_text:

        word = word.strip("?!.,:")

        if word not in vocabulary_words:

            oov.append(word)

    print("OOV words:", oov)


# ============================================================
# 11) DATA LEAKAGE
# ============================================================

print("\n========================================")
print("TASK 11 - DATA LEAKAGE")
print("========================================")

train_only_vectorizer = CountVectorizer()

train_only_vectorizer.fit(
    train_documents
)

leak_vectorizer = CountVectorizer()

leak_vectorizer.fit(
    train_documents + new_texts
)

train_vocab = (
    train_only_vectorizer
    .get_feature_names_out()
    .tolist()
)

leak_vocab = (
    leak_vectorizer
    .get_feature_names_out()
    .tolist()
)

print("Training-only vocabulary:")
print(train_vocab)

print("\nTraining + test vocabulary:")
print(leak_vocab)

print("\nWords leaked from test set:")

for word in leak_vocab:

    if word not in train_vocab:

        print(word)


# ============================================================
# 12) BAG OF WORDS DOES NOT CAPTURE MEANING
# ============================================================

print("\n========================================")
print("TASK 12 - BAG OF WORDS DOES NOT CAPTURE MEANING")
print("========================================")

review_vectorizer = CountVectorizer()

review_matrix = review_vectorizer.fit_transform(
    reviews
)

print("Vocabulary:")

print(
    review_vectorizer
    .get_feature_names_out()
    .tolist()
)

print("\nReview vectors:")

print(review_matrix.toarray())


# Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(
    review_matrix
)

print("\nCosine similarity matrix:")

print(similarity.round(2))

print("\nR1 vs R2:")
print(round(similarity[0][1], 2))

print("\nR1 vs R3:")
print(round(similarity[0][2], 2))

print("\nIs R1 more similar to R2 than R3?")

print(similarity[0][1] > similarity[0][2])


# ============================================================
# END
# ============================================================

print("\n========================================")
print("END OF ASSIGNMENT 3")
print("========================================")
