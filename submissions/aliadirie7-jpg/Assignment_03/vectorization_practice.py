from sklearn.feature_extraction.text import CountVectorizer
vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]
import numpy as np
def one_hot_encode(word):
    vector = []

    for item in vocabulary:
        if item == word:
            vector.append(1)
        else:
            vector.append(0)

    return vector


print("One-hot vector for free:")
print(one_hot_encode("free"))

print("Vector for cash:")
print(one_hot_encode("cash"))



sentence = ["win", "free", "cash"]

print("One-hot vectors for win free cash:")

for word in sentence:
    print(word, ":", one_hot_encode(word))





documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

vectorizer = CountVectorizer()

dtm = vectorizer.fit_transform(documents)

print("\n===== VOCABULARY =====")
print(vectorizer.get_feature_names_out())

print("\n===== DTM SHAPE =====")
print(dtm.shape)

print("\n===== DOCUMENT TERM MATRIX =====")
print(dtm.toarray())


test_documents = [
    "win free phone",
    "phone free win"
]
test_vectors = vectorizer.transform(test_documents)

print("\n===== TEST VECTORS =====")
print(test_vectors.toarray())

print("\n===== TASK 3 =====")

total_tokens = sum(len(doc.split()) for doc in documents)

vocabulary_size = len(vectorizer.get_feature_names_out())

print("Total tokens:", total_tokens)
print("Vocabulary size:", vocabulary_size)
print("\n===== WORDS IN MORE THAN ONE DOCUMENT =====")

import numpy as np

document_frequency = np.asarray(
    (dtm > 0).sum(axis=0)
).flatten()

feature_names = vectorizer.get_feature_names_out()

for word, df in zip(feature_names, document_frequency):
    if df > 1:
        print(word, ":", df)



D1 = documents[0]

# Default settings
cv_default = CountVectorizer()
cv_default.fit([D1])

print("===== DEFAULT VOCABULARY =====")
print(cv_default.get_feature_names_out())

cv_case = CountVectorizer(lowercase=False)
cv_case.fit([D1])

print("\n===== LOWERCASE FALSE VOCABULARY =====")
print(cv_case.get_feature_names_out())

cv_stop = CountVectorizer(stop_words="english")
cv_stop.fit([D1])

print("\n===== STOP WORDS ENGLISH VOCABULARY =====")
print(cv_stop.get_feature_names_out())

cv_all_stop = CountVectorizer(stop_words="english")
cv_all_stop.fit(documents)

print("\n===== ALL DOCUMENTS WITH ENGLISH STOP WORDS =====")
print(cv_all_stop.get_feature_names_out())

cv_min = CountVectorizer(min_df=2)
cv_min.fit(documents)

print("\n===== MIN_DF = 2 =====")
print(cv_min.get_feature_names_out())
print("Vocabulary size:", len(cv_min.get_feature_names_out()))

cv_max = CountVectorizer(max_features=5)
cv_max.fit(documents)

print("\n===== MAX_FEATURES = 5 =====")
print(cv_max.get_feature_names_out())
print("Vocabulary size:", len(cv_max.get_feature_names_out()))

D5 = documents[4]

cv_uni = CountVectorizer(ngram_range=(1, 1))
cv_uni.fit([D5])

cv_bi = CountVectorizer(ngram_range=(1, 2))
cv_bi.fit([D5])

print("\n===== UNIGRAM =====")
print(cv_uni.get_feature_names_out())

print("\n===== UNIGRAM + BIGRAM =====")
print(cv_bi.get_feature_names_out())

print("\n===== N-GRAM COMPARISON =====")

for ngram in [(1, 1), (1, 2), (1, 3)]:
    cv = CountVectorizer(ngram_range=ngram)
    cv.fit(documents)

    vocab = cv.get_feature_names_out()

    print(f"\nN-gram range: {ngram}")
    print("Vocabulary size:", len(vocab))

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n===== TF-IDF MATRIX =====")
print(tfidf_matrix.toarray().round(2))

print("\n===== TF-IDF VOCABULARY =====")
print(tfidf_vectorizer.get_feature_names_out())

feature_names = tfidf_vectorizer.get_feature_names_out()

for doc_index in [0, 1]:
    print(f"\n===== DOCUMENT {doc_index + 1} =====")

    counts = dtm[doc_index].toarray()[0]
    weights = tfidf_matrix[doc_index].toarray()[0]

    for word, count, weight in zip(feature_names, counts, weights):
        if count > 0:
            print(f"{word}: Count={count}, TF-IDF={weight:.2f}")


print("\n===== TASK 8: COUNT AND DF =====")

for word in ["cash", "today"]:
    index = list(feature_names).index(word)

    count = dtm[1, index]
    df = document_frequency[index]

    print(f"{word}: Count={count}, DF={df}")


import numpy as np

N = len(documents)

for word in ["cash", "today"]:
    index = list(feature_names).index(word)
    df = document_frequency[index]

    idf = np.log((1 + N) / (1 + df)) + 1

    print(f"{word}: IDF = {idf:.4f}")

print("\n===== MANUAL TF-IDF =====")

for word in ["cash", "today"]:
    index = list(feature_names).index(word)

    tf = dtm[1, index]
    idf = tfidf_vectorizer.idf_[index]

    raw_tfidf = tf * idf

    normalized_tfidf = tfidf_matrix[1, index]

    print(f"\nWord: {word}")
    print(f"TF: {tf}")
    print(f"IDF: {idf:.4f}")
    print(f"Raw TF-IDF: {raw_tfidf:.4f}")
    print(f"Normalized TF-IDF: {normalized_tfidf:.4f}")

print("\n===== TASK 9: SPARSE MATRIX =====")

total_cells = dtm.shape[0] * dtm.shape[1]

nonzero_cells = dtm.count_nonzero()

zero_cells = total_cells - nonzero_cells

sparsity = (zero_cells / total_cells) * 100

print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print(f"Sparsity: {sparsity:.2f}%")

print("\n===== TASK 10: OOV WORDS =====")

N1 = "Win a prize today!"
N2 = "Free cash for everyone!"

new_documents = [N1, N2]
oov_vectorizer = CountVectorizer()

oov_vectorizer.fit(documents[:4])

new_documents = [N1, N2]

new_vectors = oov_vectorizer.transform(new_documents)

print("Vocabulary:", oov_vectorizer.get_feature_names_out())
print("New document vectors:\n", new_vectors.toarray())

print("\n===== TASK 11: DATA LEAKAGE =====")

train_vectorizer = CountVectorizer()
train_vectorizer.fit(documents[:4])

all_vectorizer = CountVectorizer()
all_vectorizer.fit(documents)

print("Training vocabulary size:",
      len(train_vectorizer.get_feature_names_out()))

print("All vocabulary size:",
      len(all_vectorizer.get_feature_names_out()))

R1 = "The food is tasty"
R2 = "The food is delicious"
R3 = "The food is terrible"

reviews = [R1, R2, R3]


from sklearn.metrics.pairwise import cosine_similarity

review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(reviews)

similarity = cosine_similarity(review_matrix)

print("\n===== TASK 12: COSINE SIMILARITY =====")
print(similarity.round(2))


