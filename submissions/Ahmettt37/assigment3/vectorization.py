import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

np.set_printoptions(linewidth=200)
documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]
N1 = "Claim your free prize now"
N2 = "Is the lunch meeting moved to tomorrow?"

R1 = "the food was tasty"
R2 = "the food was delicious"
R3 = "the food was terrible"


#one-hot-encoding
vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]
print("\n One hot encoding================")
print("vocabulary: ", vocabulary)

words = "free cash".split()
for word in words:
    vector = [
        1 if vocabulary_word == word else 0 
        for vocabulary_word in vocabulary
    ]
    print(word, "->", vector)


sentsnce = "win free cash".split()
for word in sentsnce:
    vector = [
        1 if vocabulary_word == word else 0
        for vocabulary_word in vocabulary
    ]
    print(word, "->", vector)




# BAG OF WORDS

print("\n==========  BAG OF WORDS ==========")



vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(documents)
vocabulary = vectorizer.get_feature_names_out()
print("\nVocabulary:")
print(vocabulary.tolist())
print("\nShape (rows, columns):")
print(count_matrix.shape)
print("\nDocument-Term Matrix:")
print(count_matrix.toarray())

d1 = count_matrix.toarray()[0]
print("\nD1 values equal to 2:")

for index, count in enumerate(d1):
    if count == 2:
        print("Word:", vocabulary[index])
        print("Count:", count)

new_texts = [
    "win free phone",
    "phone free win"
]

new_vectors = vectorizer.transform(new_texts)

print("\nTransform new texts:")
print("win free phone:")
print(new_vectors.toarray()[0])

print("\nphone free win:")
print(new_vectors.toarray()[1])

# Compare vectors

print("\nAre both vectors the same?")

print(
    (new_vectors.toarray()[0] == new_vectors.toarray()[1]).all()
)


# 1. Total number of tokens in D1-D5


print("\n==========  tokens and vacabulary_size ==========")
total_tokens = count_matrix.sum()

print("Total tokens:", total_tokens)


# 2. Vocabulary size

vocabulary_size = len(vocabulary)

print("Vocabulary size:", vocabulary_size)


# 3. Words appearing in more than one document

document_frequency = (count_matrix > 0).toarray().sum(axis=0)

print("\nWords appearing in more than one document:")

for index, doc_count in enumerate(document_frequency):
    if doc_count > 1:
        print(vocabulary[index])


print("\n==========  processing ==========")

D1 = [documents[0]]

# 1. Default vectorizer
vectorizer1 = CountVectorizer()
vectorizer1.fit(D1)

print("Default vocabulary:")
print(vectorizer1.get_feature_names_out().tolist())


# 2. lowercase=False
vectorizer2 = CountVectorizer(lowercase=False)
vectorizer2.fit(D1)

print("\nLowercase=False vocabulary:")
print(vectorizer2.get_feature_names_out().tolist())


# 3. stop_words="english"
vectorizer3 = CountVectorizer(stop_words="english")
vectorizer3.fit(D1)

print("\nEnglish stop words vocabulary:")
print(vectorizer3.get_feature_names_out().tolist())


# 4. Fit stop_words="english" on D1-D5

vectorizer4 = CountVectorizer(stop_words="english")
vectorizer4.fit(documents)

print("\nD1-D5 vocabulary with English stop words:")
print(vectorizer4.get_feature_names_out().tolist())

# Check whether 'not' exists in vocabulary
print("\nIs 'not' in vocabulary?")
print("not" in vectorizer4.vocabulary_)




# 1. Full vocabulary size
print("\n==========  Full vocabulary size  ==========")
full_vectorizer = CountVectorizer()
full_vectorizer.fit(documents)

full_vocabulary = full_vectorizer.get_feature_names_out()

print("\nFull vocabulary:")
print(full_vocabulary.tolist())
print("Full vocabulary size:", len(full_vocabulary))


# 2. min_df=2

vectorizer_min = CountVectorizer(min_df=2)
vectorizer_min.fit(documents)

min_vocabulary = vectorizer_min.get_feature_names_out()

print("\nVocabulary with min_df=2:")
print(min_vocabulary.tolist())
print("Vocabulary size:", len(min_vocabulary))


# 3. max_features=5

vectorizer_max = CountVectorizer(max_features=5)
vectorizer_max.fit(documents)

max_vocabulary = vectorizer_max.get_feature_names_out()

print("\nVocabulary with max_features=5:")
print(max_vocabulary.tolist())
print("Vocabulary size:", len(max_vocabulary))


print("\n==========  unigram and bigram  ==========")
D5 = [documents[4]]

# Unigram: (1, 1)
vectorizer1 = CountVectorizer(ngram_range=(1, 1))
vectorizer1.fit(D5)

print("Unigram vocabulary:")
print(vectorizer1.get_feature_names_out().tolist())


# Unigram + Bigram: (1, 2)
vectorizer2 = CountVectorizer(ngram_range=(1, 2))
vectorizer2.fit(D5)

print("\nUnigram + Bigram vocabulary:")
print(vectorizer2.get_feature_names_out().tolist())



print("\n==========  TfidfVectorizer  ==========")
# TF-IDF

# 1. Create and fit TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get vocabulary
tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out()

# 2. Print TF-IDF matrix rounded to 2 decimals
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray().round(2))


# 3. Create CountVectorizer to get word counts
count_vectorizer = CountVectorizer()
count_data = count_vectorizer.fit_transform(documents)

count_vocabulary = count_vectorizer.get_feature_names_out()


# 4. Print each word, count and TF-IDF weight for D1 and D2

for doc_index in [0, 1]:

    print("\nDocument D" + str(doc_index + 1))

    counts = count_data.toarray()[doc_index]
    weights = tfidf_matrix.toarray()[doc_index]

    for index, count in enumerate(counts):

        if count > 0:
            word = count_vocabulary[index]

            print(
                "Word:", word,
                "| Count:", count,
                "| TF-IDF:", round(weights[index], 2)
            )


print("\n==========  TfidfVectorizer  ==========")

N = count_data.shape[0]

counts = count_data.toarray()[1]

total_words = counts.sum()

print("Total documents:", N)
print("Total words in D2:", total_words)

for word in ["cash", "today"]:

    index = np.where(count_vocabulary == word)[0][0]

    word_count = counts[index]

    df = (count_data.toarray()[:, index] > 0).sum()

    tf = word_count / total_words

    idf = np.log(N / df)

    tfidf = tf * idf

    print("\nWord:", word)
    print("Count:", word_count)
    print("Total words:", total_words)
    print("TF:", tf)
    print("N:", N)
    print("DF:", df)
    print("IDF:", idf)
    print("TF-IDF:", tfidf)

    print(
        "Scikit-learn TF-IDF:",
        tfidf_matrix.toarray()[1, index]
    )


    print("\n===== SPARSE MATRIX =====")

# Total number of cells
total_cells = count_data.shape[0] * count_data.shape[1]

# Number of non-zero cells
non_zero_cells = count_data.nnz

# Number of zero cells
zero_cells = total_cells - non_zero_cells

# Percentage of zeros
zero_percentage = (zero_cells / total_cells) * 100

print("Total cells:", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(zero_percentage, 2), "%")



print("\n===== FIT AND TRANSFORM =====")

# Training documents
D1 = documents[0]
D2 = documents[1]
D3 = documents[2]
D4 = documents[3]

train_documents = [D1, D2, D3, D4]

# Test documents
test_documents = [N1, N2]

# Create a new vectorizer
new_vectorizer = CountVectorizer()

# Fit and transform training documents
X_train = new_vectorizer.fit_transform(train_documents)

# Transform test documents
X_test = new_vectorizer.transform(test_documents)

# Get vocabulary
new_vocabulary = new_vectorizer.get_feature_names_out()

print("\nTraining Vocabulary:")
print(new_vocabulary)

print("\nN1 Vector:")
print(X_test.toarray()[0])

print("\nN2 Vector:")
print(X_test.toarray()[1])


print("\n===== PART 3: DATA LEAKAGE =====")

# Training documents
train = [D1, D2, D3, D4]

# Test documents
test = [N1, N2]

# Vectorizer 1: Training data only
vectorizer_train = CountVectorizer()

vectorizer_train.fit(train)

train_vocabulary = vectorizer_train.get_feature_names_out()

# Vectorizer 2: Training + Test data
vectorizer_all = CountVectorizer()

all_documents = train + test

vectorizer_all.fit(all_documents)

all_vocabulary = vectorizer_all.get_feature_names_out()

# Find leaked words
leaked_words = sorted(
    set(all_vocabulary) - set(train_vocabulary)
)

# Print results
print("\nTraining Vocabulary (D1-D4):")
print(train_vocabulary)

print("\nTraining + Test Vocabulary (D1-D4 + N1 + N2):")
print(all_vocabulary)

print("\nLeaked Words:")
print(leaked_words)


print("\n=====  NO MEANING =====")

from sklearn.metrics.pairwise import cosine_similarity

# Reviews
reviews = [R1, R2, R3]

# Create CountVectorizer
review_vectorizer = CountVectorizer()

# Fit and transform reviews
review_matrix = review_vectorizer.fit_transform(reviews)

# Get vocabulary
review_vocabulary = review_vectorizer.get_feature_names_out()

print("\nVocabulary:")
print(review_vocabulary)

# Print vectors
print("\nReview Vectors:")
print(review_matrix.toarray())

# Calculate cosine similarity
similarity_matrix = cosine_similarity(review_matrix)

print("\nCosine Similarity Matrix:")
print(similarity_matrix.round(2))