
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# train corpus documents
documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

# New documents
new_documents = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]

# Reviews
reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]


# TASK 1 - ONE-HOT ENCODING

print("\n" + "=" * 60)
print("TASK 1 - ONE-HOT ENCODING")
print("=" * 60)

vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

# Create a one-hot encoder
def one_hot(word):
    vector = []

    for vocab_word in vocabulary:
        if word == vocab_word:
            vector.append(1)
        else:
            vector.append(0)

    return vector


# Print the vector for each word in "free cash"
print("\nOne-hot vectors for each word in 'free cash':")

for word in ["free", "cash"]:
    print(word, ":", one_hot(word))


# Print one-hot vectors for the sentence "win free cash"
print("\nOne-hot vectors for 'win free cash':")

for word in ["win", "free", "cash"]:
    print(word, ":", one_hot(word))




# TASK 2 - BAG OF WORDS

print("\n" + "=" * 60)
print("TASK 2 - BAG OF WORDS")
print("=" * 60)

# Create CountVectorizer
bow_vectorizer = CountVectorizer()

# Learn vocabulary and transform documents
bow_matrix = bow_vectorizer.fit_transform(documents)

# Print vocabulary
print("\nVocabulary:")
print(bow_vectorizer.get_feature_names_out())

# Print matrix shape
print("\nMatrix Shape:")
print(bow_matrix.shape)

# Print full document-term matrix
print("\nDocument-Term Matrix:")
print(bow_matrix.toarray())

# Transform two new sentences
test_sentences = [
    "win free phone",
    "phone free win"
]

test_vectors = bow_vectorizer.transform(test_sentences)

print("\nTest Sentence Vectors:")
for sentence, vector in zip(test_sentences, test_vectors.toarray()):
    print(sentence, ":", vector)



# TASK 3 - TOKENS VS VOCABULARY SIZE

print("\n" + "=" * 60)
print("TASK 3 - TOKENS VS VOCABULARY SIZE")
print("=" * 60)

# Use CountVectorizer's analyzer
analyzer = bow_vectorizer.build_analyzer()

# Count total tokens
total_tokens = sum(len(analyzer(document)) for document in documents)

# Get vocabulary size
vocabulary_size = len(bow_vectorizer.get_feature_names_out())

print("\nTotal Tokens:")
print(total_tokens)

print("\nVocabulary Size:")
print(vocabulary_size)

# Find words appearing in more than one document
word_document_counts = {}

for document in documents:
    unique_words = set(analyzer(document))

    for word in unique_words:
        word_document_counts[word] = word_document_counts.get(word, 0) + 1

repeated_words = {
    word: count
    for word, count in word_document_counts.items()
    if count > 1
}

print("\nWords appearing in more than one document:")
print(repeated_words)


# TASK 4 - PREPROCESSING INSIDE THE VECTORIZER

print("\n" + "=" * 60)
print("TASK 4 - PREPROCESSING INSIDE THE VECTORIZER")
print("=" * 60)

# Default vectorizer on D1
default_vectorizer = CountVectorizer()
default_vectorizer.fit([documents[0]])

print("\nDefault Vocabulary:")
print(default_vectorizer.get_feature_names_out())

# lowercase=False
case_vectorizer = CountVectorizer(lowercase=False)
case_vectorizer.fit([documents[0]])

print("\nVocabulary with lowercase=False:")
print(case_vectorizer.get_feature_names_out())

# English stop words on D1
stop_vectorizer = CountVectorizer(stop_words="english")
stop_vectorizer.fit([documents[0]])

print("\nVocabulary with stop_words='english' on D1:")
print(stop_vectorizer.get_feature_names_out())

# English stop words on D1-D5
all_stop_vectorizer = CountVectorizer(stop_words="english")
all_stop_vectorizer.fit(documents)

print("\nVocabulary with stop_words='english' on D1-D5:")
print(all_stop_vectorizer.get_feature_names_out())

print("\nIs 'not' in the vocabulary?")
print("not" in all_stop_vectorizer.vocabulary_)



# TASK 5 - LIMITING THE VOCABULARY

print("\n" + "=" * 60)
print("TASK 5 - LIMITING THE VOCABULARY")
print("=" * 60)

# Full vocabulary size
full_vocab = CountVectorizer()
full_vocab.fit(documents)

full_size = len(full_vocab.get_feature_names_out())

print("\nFull Vocabulary:")
print(full_vocab.get_feature_names_out())
print("Full Vocabulary Size:", full_size)

# min_df = 2
min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

min_df_vocab = min_df_vectorizer.get_feature_names_out()

print("\nVocabulary with min_df=2:")
print(min_df_vocab)
print("Vocabulary Size:", len(min_df_vocab))

# max_features = 5
max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

max_features_vocab = max_features_vectorizer.get_feature_names_out()

print("\nVocabulary with max_features=5:")
print(max_features_vocab)
print("Vocabulary Size:", len(max_features_vocab))


# TASK 6 - N-GRAMS
print("\n" + "=" * 60)
print("TASK 6 - N-GRAMS")
print("=" * 60)

# Unigrams from D5
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit([documents[4]])

print("\nD5 Unigram Vocabulary:")
print(unigram_vectorizer.get_feature_names_out())

# Unigrams and bigrams from D5
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit([documents[4]])

print("\nD5 Unigram + Bigram Vocabulary:")
print(bigram_vectorizer.get_feature_names_out())

# Compare vocabulary sizes across all documents
for ngram in [(1, 1), (1, 2), (1, 3)]:
    vectorizer = CountVectorizer(ngram_range=ngram)
    vectorizer.fit(documents)

    print(f"\nN-gram range {ngram}:")
    print("Vocabulary Size:", len(vectorizer.get_feature_names_out()))


    # TASK 7 - TF-IDF
print("\n" + "=" * 60)
print("TASK 7 - TF-IDF")
print("=" * 60)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

terms = tfidf_vectorizer.get_feature_names_out()

print("\nTF-IDF Matrix:")
print(np.round(tfidf_matrix.toarray(), 2))

# Show counts and TF-IDF weights for D1 and D2
for doc_index, doc_name in [(0, "D1"), (1, "D2")]:
    count_row = bow_matrix.getrow(doc_index).toarray().ravel()
    tfidf_row = tfidf_matrix.getrow(doc_index).toarray().ravel()

    print(f"\n{doc_name} - Word Counts and TF-IDF Weights:")
    for term, count, weight in zip(terms, count_row, tfidf_row):
        if count > 0:
            print(term, "| Count:", count, "| TF-IDF:", round(weight, 2))


            
        # TASK 8 - TF-IDF BY HAND
print("\n" + "=" * 60)
print("TASK 8 - TF-IDF BY HAND")
print("=" * 60)

analyzer = bow_vectorizer.build_analyzer()
d2_tokens = analyzer(documents[1])

N = len(documents)

for word in ["cash", "today"]:
    count = d2_tokens.count(word)

    df = sum(
        1 for doc in documents
        if word in analyzer(doc)
    )

    tf = count / len(d2_tokens)
    idf = np.log(N / df)
    tfidf = tf * idf

    print(f"\nWord: {word}")
    print("Count:", count)
    print("Total tokens in D2:", len(d2_tokens))
    print("Document Frequency:", df)
    print("TF:", round(tf, 4))
    print("IDF:", round(idf, 4))
    print("TF-IDF:", round(tfidf, 4))


    # TASK 9 - SPARSE MATRIX
print("\n" + "=" * 60)
print("TASK 9 - SPARSE MATRIX")
print("=" * 60)

total_cells = bow_matrix.shape[0] * bow_matrix.shape[1]

nonzero_values = bow_matrix.nnz

zero_values = total_cells - nonzero_values

zero_percentage = (zero_values / total_cells) * 100

print("\nTotal Cells:", total_cells)
print("Non-zero Values:", nonzero_values)
print("Zero Values:", zero_values)
print("Percentage of Zeros:", round(zero_percentage, 2), "%")

# TASK 10 - FIT VS TRANSFORM
print("\n" + "=" * 60)
print("TASK 10 - FIT VS TRANSFORM")
print("=" * 60)

train_documents = documents[:4]

fit_vectorizer = CountVectorizer()

train_matrix = fit_vectorizer.fit_transform(train_documents)

print("\nTraining Vocabulary:")
print(fit_vectorizer.get_feature_names_out())

test_matrix = fit_vectorizer.transform(new_documents)

print("\nNew Document Vectors:")

analyzer = fit_vectorizer.build_analyzer()
vocabulary = fit_vectorizer.vocabulary_

for text, vector in zip(new_documents, test_matrix.toarray()):
    print("\nText:", text)
    print("Vector:", vector)

    tokens = analyzer(text)

    oov_words = [
        word for word in tokens
        if word not in vocabulary
    ]

    print("OOV Words:", oov_words)


    # TASK 11 - DATA LEAKAGE
print("\n" + "=" * 60)
print("TASK 11 - DATA LEAKAGE")
print("=" * 60)

# Fit on training documents only
train_vectorizer = CountVectorizer()
train_vectorizer.fit(train_documents)

train_vocab = set(train_vectorizer.get_feature_names_out())

print("\nTraining Vocabulary:")
print(sorted(train_vocab))

# Fit on training and test documents together
combined_vectorizer = CountVectorizer()
combined_vectorizer.fit(train_documents + new_documents)

combined_vocab = set(combined_vectorizer.get_feature_names_out())

print("\nCombined Vocabulary:")
print(sorted(combined_vocab))

# Find words that entered the vocabulary from test documents
test_analyzer = combined_vectorizer.build_analyzer()

test_words = set()
for text in new_documents:
    test_words.update(test_analyzer(text))

leaked_words = sorted(
    (combined_vocab - train_vocab) & test_words
)

print("\nWords Leaked from Test Set:")
print(leaked_words)



# TASK 12 - VECTORIZATION DOES NOT UNDERSTAND MEANING
print("\n" + "=" * 60)
print("TASK 12 - VECTORIZATION DOES NOT UNDERSTAND MEANING")
print("=" * 60)

review_vectorizer = CountVectorizer()

review_matrix = review_vectorizer.fit_transform(reviews)

print("\nReview Vocabulary:")
print(review_vectorizer.get_feature_names_out())

print("\nReview Vectors:")
print(review_matrix.toarray())

similarity_matrix = cosine_similarity(review_matrix)

print("\nCosine Similarity Matrix:")
print(np.round(similarity_matrix, 2))

print("\nR1 vs R2:", round(similarity_matrix[0][1], 2))
print("R1 vs R3:", round(similarity_matrix[0][2], 2))



