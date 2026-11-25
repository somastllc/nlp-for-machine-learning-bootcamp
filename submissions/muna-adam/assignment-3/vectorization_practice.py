import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]


# Test documents
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

# ============================================================
# TASK 1: ONE-HOT ENCODING
# ============================================================

print("\n" + "=" * 60)
print("TASK 1: ONE-HOT ENCODING")
print("=" * 60)

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]


word_positions = {
    word: index for index, word in enumerate(vocabulary)
}

def one_hot_encode(word):
    vector = [0] * len(vocabulary)

    if word in word_positions:
        vector[word_positions[word]] = 1

    return vector


print("\nOne-hot vectors for 'free cash':")

for word in ["free", "cash"]:
    print(word, ":", one_hot_encode(word))

print("\nOne-hot vectors for 'win free cash':")

for word in ["win", "free", "cash"]:
    print(word, ":", one_hot_encode(word))


# ============================================================
# TASK 2: BAG OF WORDS
# ============================================================

print("\n" + "=" * 60)
print("TASK 2: BAG OF WORDS")
print("=" * 60)


count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)


print("\nVocabulary:")
print(count_vectorizer.get_feature_names_out())


print("\nDocument-Term Matrix Shape:")
print(count_matrix.shape)


print("\nFull Document-Term Matrix:")
print(count_matrix.toarray())


feature_names = count_vectorizer.get_feature_names_out()
d1_row = count_matrix.toarray()[0]

print("\nWords with count 2 in D1:")

for index, count in enumerate(d1_row):
    if count == 2:
        print(feature_names[index], ":", count)


test_sentences = ["win free phone", "phone free win"]
test_vectors = count_vectorizer.transform(test_sentences)

print("\nVector for 'win free phone':")
print(test_vectors.toarray()[0])

print("\nVector for 'phone free win':")
print(test_vectors.toarray()[1])


# ============================================================
# TASK 3: TOKENS VS VOCABULARY SIZE
# ============================================================

print("\n" + "=" * 60)
print("TASK 3: TOKENS VS VOCABULARY SIZE")
print("=" * 60)

all_tokens = []

for document in documents:
    tokens = document.lower().split()
    all_tokens.extend(tokens)

print("\nTotal tokens (simple whitespace split):")
print(len(all_tokens))


print("\nVocabulary size:")
print(len(count_vectorizer.get_feature_names_out()))


document_frequency = Counter()

for document in documents:
    unique_words = set(
        CountVectorizer().build_analyzer()(document)
    )

    document_frequency.update(unique_words)

print("\nWords appearing in more than one document:")

for word, frequency in sorted(document_frequency.items()):
    if frequency > 1:
        print(word, ":", frequency, "documents")


# ============================================================
# TASK 4: PREPROCESSING INSIDE THE VECTORIZER
# ============================================================

print("\n" + "=" * 60)
print("TASK 4: PREPROCESSING INSIDE THE VECTORIZER")
print("=" * 60)


d1 = documents[0]


default_vectorizer = CountVectorizer()
default_vectorizer.fit([d1])

print("\nDefault vocabulary:")
print(default_vectorizer.get_feature_names_out())


no_lowercase_vectorizer = CountVectorizer(lowercase=False)
no_lowercase_vectorizer.fit([d1])

print("\nVocabulary with lowercase=False:")
print(no_lowercase_vectorizer.get_feature_names_out())

stopword_vectorizer = CountVectorizer(stop_words="english")
stopword_vectorizer.fit([d1])

print("\nVocabulary with stop_words='english':")
print(stopword_vectorizer.get_feature_names_out())


all_stopword_vectorizer = CountVectorizer(stop_words="english")
all_stopword_vectorizer.fit(documents)

print("\nVocabulary without stop words on D1-D5:")
print(all_stopword_vectorizer.get_feature_names_out())

print("\nIs 'not' in the vocabulary?")
print("not" in all_stopword_vectorizer.vocabulary_)

# ============================================================
# TASK 5: LIMITING THE VOCABULARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 5: LIMITING THE VOCABULARY")
print("=" * 60)


min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

min_df_vocab = min_df_vectorizer.get_feature_names_out()

print("\nVocabulary with min_df=2:")
print(min_df_vocab)
print("Vocabulary size:", len(min_df_vocab))


max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

max_features_vocab = max_features_vectorizer.get_feature_names_out()

print("\nVocabulary with max_features=5:")
print(max_features_vocab)
print("Vocabulary size:", len(max_features_vocab))

# ============================================================
# TASK 6: N-GRAMS
# ============================================================

print("\n" + "=" * 60)
print("TASK 6: N-GRAMS")
print("=" * 60)

# Use D5 only
d5 = documents[4]


unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit([d5])

print("\nD5 Unigram vocabulary:")
print(unigram_vectorizer.get_feature_names_out())


bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit([d5])

print("\nD5 Unigram + Bigram vocabulary:")
print(bigram_vectorizer.get_feature_names_out())


print("\nN-gram vocabulary sizes on D1-D5:")

for ngram in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram)
    vectorizer.fit(documents)

    print(
        "ngram_range =", ngram,
        "| Vocabulary size =", len(vectorizer.vocabulary_)
    )

# ============================================================
# TASK 7: TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("TASK 7: TF-IDF")
print("=" * 60)


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_features = tfidf_vectorizer.get_feature_names_out()


print("\nTF-IDF Matrix:")
print(np.round(tfidf_matrix.toarray(), 2))


count_array = count_matrix.toarray()
tfidf_array = tfidf_matrix.toarray()

for document_index, document_name in [(0, "D1"), (1, "D2")]:

    print("\n", document_name, "- Word Counts and TF-IDF Weights")

    for index, word in enumerate(tfidf_features):
        count = count_array[document_index, index]
        weight = tfidf_array[document_index, index]

        if count > 0:
            print(
                f"{word}: Count = {count}, TF-IDF = {weight:.4f}"
            )

# ============================================================
# TASK 8: TF-IDF BY HAND
# ============================================================

print("\n" + "=" * 60)
print("TASK 8: TF-IDF BY HAND")
print("=" * 60)


N = len(documents)
d2_tokens = CountVectorizer().build_analyzer()(documents[1])
d2_counts = Counter(d2_tokens)


df_values = np.asarray((count_matrix > 0).sum(axis=0)).ravel()

for word in ["cash", "today"]:


    count = d2_counts[word]

    
    tf = count

  
    word_index = count_vectorizer.vocabulary_[word]
    df = int(df_values[word_index])

    
    idf = np.log(N / df)

   
    tfidf_hand = tf * idf

    print(f"\nWord: {word}")
    print("Count in D2:", count)
    print("TF:", tf)
    print("Number of documents (N):", N)
    print("Document frequency (df):", df)
    print("IDF = log(N / df):", idf)
    print("TF-IDF = TF * IDF:", tfidf_hand)

# ============================================================
# TASK 9: SPARSE MATRIX
# ============================================================

print("\n" + "=" * 60)
print("TASK 9: SPARSE MATRIX")
print("=" * 60)

rows, columns = count_matrix.shape


total_cells = rows * columns


nonzero_cells = count_matrix.nnz

zero_cells = total_cells - nonzero_cells


zero_percentage = (zero_cells / total_cells) * 100

print("\nTotal cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print(f"Percentage of zeros: {zero_percentage:.2f}%")

# ============================================================
# TASK 10: FIT AND TRANSFORM
# ============================================================

print("\n" + "=" * 60)
print("TASK 10: FIT AND TRANSFORM")
print("=" * 60)


training_documents = documents[:4]

training_vectorizer = CountVectorizer()
training_matrix = training_vectorizer.fit_transform(training_documents)

print("\nTraining vocabulary:")
print(training_vectorizer.get_feature_names_out())

new_vectors = training_vectorizer.transform(new_texts)

print("\nN1 Vector:")
print(new_vectors.toarray()[0])

print("\nN2 Vector:")
print(new_vectors.toarray()[1])


analyzer = training_vectorizer.build_analyzer()
training_vocabulary = set(training_vectorizer.get_feature_names_out())

for name, text in zip(["N1", "N2"], new_texts):

    test_words = set(analyzer(text))
    unknown_words = sorted(test_words - training_vocabulary)

    print(f"\n{name} out-of-vocabulary words:")
    print(unknown_words)



# ============================================================
# TASK 11: DATA LEAKAGE
# ============================================================

print("\n" + "=" * 60)
print("TASK 11: DATA LEAKAGE")
print("=" * 60)

train_only_vectorizer = CountVectorizer()
train_only_vectorizer.fit(training_documents)


combined_documents = training_documents + new_texts

combined_vectorizer = CountVectorizer()
combined_vectorizer.fit(combined_documents)


train_vocab = set(train_only_vectorizer.get_feature_names_out())
combined_vocab = set(combined_vectorizer.get_feature_names_out())

print("\nTraining-only vocabulary:")
print(sorted(train_vocab))

print("\nTraining + Test vocabulary:")
print(sorted(combined_vocab))


leaked_words = sorted(combined_vocab - train_vocab)

print("\nWords leaked from the test set:")
print(leaked_words)

# ============================================================
# TASK 12: NO MEANING - COSINE SIMILARITY
# ============================================================

print("\n" + "=" * 60)
print("TASK 12: NO MEANING - COSINE SIMILARITY")
print("=" * 60)

review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(reviews)

print("\nReview vocabulary:")
print(review_vectorizer.get_feature_names_out())


print("\nReview vectors:")
print(review_matrix.toarray())

similarity_matrix = cosine_similarity(review_matrix)

print("\nCosine Similarity Matrix:")
print(np.round(similarity_matrix, 4))

r1_r2 = similarity_matrix[0, 1]
r1_r3 = similarity_matrix[0, 2]

print("\nR1 vs R2 similarity:", r1_r2)
print("R1 vs R3 similarity:", r1_r3)

print("\nIs R1 more similar to R2 than R3?")

if r1_r2 > r1_r3:
    print("Yes")
elif r1_r2 < r1_r3:
    print("No")
else:
    print("They are equally similar.")








