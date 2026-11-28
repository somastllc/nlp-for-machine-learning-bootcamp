import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=200)  # print each matrix row on one line

documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great.",
]

# --------------------------------
# 1) One-hot encoding
# --------------------------------

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

print("\n=== 1) ONE-HOT ENCODING ===")
print("Vocabulary:", vocabulary)

for word in ["free", "cash", "win", "free", "cash"]:
    vector = []

    for item in vocabulary:
        if word == item:
            vector.append(1)
        else:
            vector.append(0)

    print(word, "->", vector)



# --------------------------------
# 2) Bag of Words and document-term matrix
# Count how many times each word appears in each document.
# --------------------------------

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print("\n=== 2) BAG OF WORDS ===")
print("Vocabulary:", count_vectorizer.get_feature_names_out().tolist())
print("Shape:", count_matrix.shape)
print("Document-Term Matrix:")
print(count_matrix.toarray())

print("'win free phone':", count_vectorizer.transform(["win free phone"]).toarray())
print("'phone free win':", count_vectorizer.transform(["phone free win"]).toarray())


# --------------------------------
# 3) Tokens vs vocabulary size
# Tokens count every word occurrence; vocabulary counts unique words.
# --------------------------------

all_tokens = " ".join(documents).split()
vocabulary_words = count_vectorizer.get_feature_names_out()
document_matrix = count_matrix.toarray()

print("\n=== 3) TOKENS vs VOCABULARY SIZE ===")
print("Total tokens:", len(all_tokens))
print("Vocabulary size:", len(vocabulary_words))

print("Words appearing in more than one document:")

for index, word in enumerate(vocabulary_words):
    document_count = sum(row[index] > 0 for row in document_matrix)

    if document_count > 1:
        print(word)



# --------------------------------
# 4) Preprocessing inside the vectorizer
# The vectorizer lowercases text and removes punctuation by default.
# --------------------------------

message = [documents[0]]

print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")
print("Message:", message[0])

print("Default:",
      CountVectorizer().fit(message).get_feature_names_out().tolist())

print("lowercase=False:",
      CountVectorizer(lowercase=False).fit(message).get_feature_names_out().tolist())

print("stop_words='english':",
      CountVectorizer(stop_words="english").fit(message).get_feature_names_out().tolist())

print("\nChecking stopwords in all documents:")

stop_vectorizer = CountVectorizer(stop_words="english")
stop_vectorizer.fit(documents)

print("Vocabulary:", stop_vectorizer.get_feature_names_out().tolist())
print("'not' in vocabulary:", "not" in stop_vectorizer.get_feature_names_out())



# --------------------------------
# 5) Limiting the vocabulary
# Reduce the number of words in the vocabulary.
# --------------------------------

min_vectorizer = CountVectorizer(min_df=2)
min_vectorizer.fit(documents)

max_vectorizer = CountVectorizer(max_features=5)
max_vectorizer.fit(documents)

print("\n=== 5) LIMITING THE VOCABULARY ===")

print("Full vocabulary size:",
      len(count_vectorizer.get_feature_names_out()))

print("min_df=2:",
      min_vectorizer.get_feature_names_out().tolist())

print("min_df size:",
      len(min_vectorizer.get_feature_names_out()))

print("max_features=5:",
      max_vectorizer.get_feature_names_out().tolist())

print("max_features size:",
      len(max_vectorizer.get_feature_names_out()))



# --------------------------------
# 6) N-grams
# N-grams capture groups of neighboring words.
# --------------------------------

sentence = [documents[4]]

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit(sentence)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit(sentence)

print("\n=== 6) N-GRAMS ===")
print("Sentence:", sentence[0])

print("Unigrams:",
      unigram_vectorizer.get_feature_names_out().tolist())

print("Unigrams and bigrams:",
      bigram_vectorizer.get_feature_names_out().tolist())

print("\nVocabulary sizes on all documents:")

for n in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=n)
    vectorizer.fit(documents)

    print(n, "->", len(vectorizer.get_feature_names_out()))


# --------------------------------
# 7) TF-IDF
# Calculate the importance of each word in the documents.
# --------------------------------

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n=== 7) TF-IDF ===")
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray().round(2))

print("\nCount vs TF-IDF for D1 and D2:")

vocabulary = count_vectorizer.get_feature_names_out()
counts = count_matrix.toarray()
weights = tfidf_matrix.toarray()

for i in [0, 1]:
    print("\nDocument:", "D" + str(i + 1))

    for word, count, weight in zip(vocabulary, counts[i], weights[i]):
        if count > 0:
            print(word, "-> count:", count, "| tf-idf:", round(weight, 3))




# --------------------------------
# 7) TF-IDF
# Calculate the importance of each word in the documents.
# --------------------------------

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n=== 7) TF-IDF ===")
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray().round(2))

vocabulary = count_vectorizer.get_feature_names_out()
counts = count_matrix.toarray()
weights = tfidf_matrix.toarray()

for i in [0, 1]:
    print("\nCount vs TF-IDF for D" + str(i + 1))

    for word, count, weight in zip(vocabulary, counts[i], weights[i]):
        if count > 0:
            print(word, "-> count:", count, "| tf-idf:", round(weight, 3))


# --------------------------------
# 8) TF-IDF by hand
# TF = count / words in document
# IDF = log(N / df)
# TF-IDF = TF * IDF
# --------------------------------

N = len(documents)
words_in_document = len(documents[1].split())

vocabulary = count_vectorizer.get_feature_names_out()
counts = count_matrix.toarray()

print("\n=== 8) TF-IDF BY HAND FOR D2 ===")

for word in ["cash", "today"]:
    index = list(vocabulary).index(word)

    count = counts[1][index]
    df = sum(counts[:, index] > 0)

    tf = count / words_in_document
    idf = np.log(N / df)
    tfidf = tf * idf

    print("\nWord:", word)
    print("Count:", count)
    print("Words in D2:", words_in_document)
    print("N:", N)
    print("DF:", df)
    print("TF:", round(tf, 4))
    print("IDF:", round(idf, 4))
    print("TF-IDF:", round(tfidf, 4))



# --------------------------------
# 9) Sparse matrix
# Count zero and non-zero cells in the document-term matrix.
# --------------------------------

total_cells = count_matrix.shape[0] * count_matrix.shape[1]
non_zero_cells = count_matrix.nnz
zero_cells = total_cells - non_zero_cells
zero_percentage = (zero_cells / total_cells) * 100

print("\n=== 9) SPARSE MATRIX ===")
print("Total cells:", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(zero_percentage, 2), "%")



# --------------------------------
# 10) Fit on training text, transform new text
# Fit learns the vocabulary; transform uses the learned vocabulary.
# --------------------------------

train_documents = documents[:4]

test_documents = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_documents)
X_test = vectorizer.transform(test_documents)

X_train_vectorizer = CountVectorizer()
X_train = X_train_vectorizer.fit_transform(train_documents)
print("\n=== 10) FIT ON TRAIN, TRANSFORM NEW TEXT ===")
print("Training vocabulary:",
      vectorizer.get_feature_names_out().tolist())

print("\nTest vectors:")
print("N1:", X_test.toarray()[0])
print("N2:", X_test.toarray()[1])

print("\nOut-of-vocabulary words:")

for text in test_documents:
    words = vectorizer.build_analyzer()(text)
    unknown_words = [
        word for word in words
        if word not in vectorizer.vocabulary_
    ]

    print(text, "->", unknown_words)






# 11) DATA LEAKAGE

leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(train_documents + test_documents)

print("\n=== 11) DATA LEAKAGE ===")

print("Training vocabulary:")
print(X_train_vectorizer.get_feature_names_out().tolist())

print("\nVocabulary with test data:")
print(leaky_vectorizer.get_feature_names_out().tolist())

leaked_words = sorted(
    set(leaky_vectorizer.get_feature_names_out()) -
    set(X_train_vectorizer.get_feature_names_out())
)

print("\nLeaked words:")
print(leaked_words)



# 12) BAG OF WORDS CANNOT UNDERSTAND MEANING

reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]

review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(reviews)

similarity_matrix = cosine_similarity(review_matrix)

print("\n=== 12) BAG OF WORDS AND MEANING ===")

print("Vocabulary:")
print(review_vectorizer.get_feature_names_out().tolist())

print("\nReview Vectors:")
print(review_matrix.toarray())

print("\nCosine Similarity:")
print(similarity_matrix.round(2))



