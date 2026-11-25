import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    "the food was terrisble"
]


# ============================================================
# 1. ONE-HOT ENCODING
# ============================================================

print("=" * 60)
print("1. ONE-HOT ENCODING")
print("=" * 60)

one_hot_vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

print("Vocabulary:")
print(one_hot_vocabulary)

print()
print("One-hot vectors for free cash:")

for word in "free cash".split():
    vector = []

    for vocabulary_word in one_hot_vocabulary:
        if word == vocabulary_word:
            vector.append(1)
        else:
            vector.append(0)

    print(word, "->", vector)

print()
print("One-hot vectors for win free cash:")

for word in "win free cash".split():
    vector = []

    for vocabulary_word in one_hot_vocabulary:
        if word == vocabulary_word:
            vector.append(1)
        else:
            vector.append(0)

    print(word, "->", vector)

print()
print("The number of one-hot vectors depends on the number of words")
print("in the sentence because each word gets its own vector.")


# ============================================================
# 2. BAG OF WORDS
# ============================================================

print()
print("=" * 60)
print("2. BAG OF WORDS")
print("=" * 60)

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print("Vocabulary:")
print(count_vectorizer.get_feature_names_out().tolist())

print()
print("Shape:")
print(count_matrix.shape)

print()
print("Document-term matrix:")
print(count_matrix.toarray())

print()
print("D1 has value 2 for 'win' because WIN appears twice in D1.")

print()
print("'win free phone':")
print(count_vectorizer.transform(["win free phone"]).toarray())

print()
print("'phone free win':")
print(count_vectorizer.transform(["phone free win"]).toarray())

print()
print("The two vectors are the same because Bag of Words does not keep word order.")


# ============================================================
# 3. TOKENS VS VOCABULARY SIZE
# ============================================================

print()
print("=" * 60)
print("3. TOKENS VS VOCABULARY SIZE")
print("=" * 60)

all_tokens = count_vectorizer.build_analyzer()

total_tokens = 0

for document in documents:
    total_tokens += len(all_tokens(document))

print("Total tokens:", total_tokens)

vocabulary = count_vectorizer.get_feature_names_out()

print("Vocabulary size:", len(vocabulary))

print()
print("Words appearing in more than one document:")

for word in vocabulary:
    document_count = 0

    for document in documents:
        if word in all_tokens(document):
            document_count += 1

    if document_count > 1:
        print(word, "->", document_count, "documents")


# ============================================================
# 4. PREPROCESSING INSIDE THE VECTORIZER
# ============================================================

print()
print("=" * 60)
print("4. PREPROCESSING INSIDE THE VECTORIZER")
print("=" * 60)

d1 = documents[0]

print("D1:")
print(d1)

default_vectorizer = CountVectorizer()
default_vectorizer.fit([d1])

print()
print("Default:")
print(default_vectorizer.get_feature_names_out().tolist())

lowercase_false_vectorizer = CountVectorizer(lowercase=False)
lowercase_false_vectorizer.fit([d1])

print()
print("lowercase=False:")
print(lowercase_false_vectorizer.get_feature_names_out().tolist())

stopword_vectorizer = CountVectorizer(stop_words="english")
stopword_vectorizer.fit([d1])

print()
print("stop_words='english':")
print(stopword_vectorizer.get_feature_names_out().tolist())

print()
print("The punctuation ! is removed by the default tokenization.")
print("The one-letter word 'a' is not included.")
print("With lowercase=False, WIN and Win remain different.")
print("With the default settings, WIN and Win become the same word: win.")

print()
print("Words removed by the English stopword list:")
print("a, now, to")

stopword_all_vectorizer = CountVectorizer(stop_words="english")
stopword_all_vectorizer.fit(documents)

print()
print("D1-D5 with stop_words='english':")
print(stopword_all_vectorizer.get_feature_names_out().tolist())

print()
print("'not' was removed.")
print("This means the negative meaning in D5 can be lost if English stopwords are used.")


# ============================================================
# 5. LIMITING THE VOCABULARY
# ============================================================

print()
print("=" * 60)
print("5. LIMITING THE VOCABULARY")
print("=" * 60)

full_vectorizer = CountVectorizer()
full_vectorizer.fit(documents)

min_df_vectorizer = CountVectorizer(min_df=2)
min_df_vectorizer.fit(documents)

max_features_vectorizer = CountVectorizer(max_features=5)
max_features_vectorizer.fit(documents)

print("Full vocabulary:")
print(full_vectorizer.get_feature_names_out().tolist())

print("Full vocabulary size:")
print(len(full_vectorizer.get_feature_names_out()))

print()
print("min_df=2:")
print(min_df_vectorizer.get_feature_names_out().tolist())

print("min_df=2 size:")
print(len(min_df_vectorizer.get_feature_names_out()))

print()
print("max_features=5:")
print(max_features_vectorizer.get_feature_names_out().tolist())

print("max_features=5 size:")
print(len(max_features_vectorizer.get_feature_names_out()))


# ============================================================
# 6. N-GRAMS
# ============================================================

print()
print("=" * 60)
print("6. N-GRAMS")
print("=" * 60)

d5 = documents[4]

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit([d5])

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit([d5])

print("D5:")
print(d5)

print()
print("Unigrams:")
print(unigram_vectorizer.get_feature_names_out().tolist())

print()
print("Unigrams and bigrams:")
print(bigram_vectorizer.get_feature_names_out().tolist())

print()
print("The bigram 'not good' keeps the reviewer's negative opinion.")

unigram_all = CountVectorizer(ngram_range=(1, 1))
bigram_all = CountVectorizer(ngram_range=(1, 2))
trigram_all = CountVectorizer(ngram_range=(1, 3))

unigram_all.fit(documents)
bigram_all.fit(documents)
trigram_all.fit(documents)

print()
print("Vocabulary sizes on D1-D5:")
print("Unigrams:", len(unigram_all.get_feature_names_out()))
print("Unigrams + bigrams:", len(bigram_all.get_feature_names_out()))
print("Unigrams + bigrams + trigrams:", len(trigram_all.get_feature_names_out()))


# ============================================================
# 7. TF-IDF
# ============================================================

print()
print("=" * 60)
print("7. TF-IDF")
print("=" * 60)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out()

print("TF-IDF matrix:")
print(np.round(tfidf_matrix.toarray(), 2))

print()
print("D1 counts and TF-IDF:")

d1_counts = count_matrix.toarray()[0]
d1_tfidf = tfidf_matrix.toarray()[0]

for word, count, weight in zip(vocabulary, d1_counts, d1_tfidf):
    if count > 0:
        print(word, "-> count:", count, "| tf-idf:", round(weight, 3))

print()
print("D2 counts and TF-IDF:")

d2_counts = count_matrix.toarray()[1]
d2_tfidf = tfidf_matrix.toarray()[1]

for word, count, weight in zip(vocabulary, d2_counts, d2_tfidf):
    if count > 0:
        print(word, "-> count:", count, "| tf-idf:", round(weight, 3))

print()
print("In D2 every word appears once, but words that occur in fewer")
print("documents receive higher TF-IDF weights.")

print()
print("In D1, win appears twice, which gives it a higher TF component.")
print("It therefore receives the highest TF-IDF weight in D1.")


# ============================================================
# 8. TF-IDF BY HAND
# ============================================================

print()
print("=" * 60)
print("8. TF-IDF BY HAND")
print("=" * 60)

N = len(documents)
d2_words = count_vectorizer.build_analyzer()(documents[1])
d2_word_count = len(d2_words)

print("D2 tokens:")
print(d2_words)

print()
print("Number of words in D2:", d2_word_count)

for word in ["cash", "today"]:

    count = 0

    for token in d2_words:
        if token == word:
            count += 1

    df = 0

    for document in documents:
        if word in count_vectorizer.build_analyzer()(document):
            df += 1

    tf = count / d2_word_count
    idf = np.log(N / df)
    tf_idf = tf * idf

    print()
    print(word)
    print("Count:", count)
    print("df:", df)
    print("TF:", tf)
    print("IDF:", idf)
    print("TF-IDF:", tf_idf)

print()
print("The hand calculation gives the same ranking for cash and today")
print("because both have the same count and the same document frequency.")


# ============================================================
# 9. SPARSE MATRIX
# ============================================================

print()
print("=" * 60)
print("9. SPARSE MATRIX")
print("=" * 60)

rows = count_matrix.shape[0]
columns = count_matrix.shape[1]

total_cells = rows * columns
non_zero_cells = count_matrix.nnz
zero_cells = total_cells - non_zero_cells
zero_percentage = (zero_cells / total_cells) * 100

print("Rows:", rows)
print("Columns:", columns)
print("Total cells:", total_cells)
print("Non-zero cells:", non_zero_cells)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(zero_percentage, 2), "%")


# ============================================================
# 10. FIT AND TRANSFORM
# ============================================================

print()
print("=" * 60)
print("10. FIT AND TRANSFORM")
print("=" * 60)

training_documents = documents[:4]

training_vectorizer = CountVectorizer()
training_vectorizer.fit(training_documents)

print("Training vocabulary:")
print(training_vectorizer.get_feature_names_out().tolist())

print()
print("N1:")
print(new_texts[0])
print(training_vectorizer.transform([new_texts[0]]).toarray())

print()
print("N2:")
print(new_texts[1])
print(training_vectorizer.transform([new_texts[1]]).toarray())

training_vocabulary = set(training_vectorizer.get_feature_names_out())

print()
print("Out-of-vocabulary words in N1:")

n1_tokens = training_vectorizer.build_analyzer()(new_texts[0])

for word in n1_tokens:
    if word not in training_vocabulary:
        print(word)

print()
print("Out-of-vocabulary words in N2:")

n2_tokens = training_vectorizer.build_analyzer()(new_texts[1])

for word in n2_tokens:
    if word not in training_vocabulary:
        print(word)

print()
print("'prize' is unknown because the training data contains 'prizes',")
print("which is a different token.")

print()
print("Stemming or lemmatization from Assignment 2 could help combine")
print("related word forms before vectorization.")


# ============================================================
# 11. DATA LEAKAGE
# ============================================================

print()
print("=" * 60)
print("11. DATA LEAKAGE")
print("=" * 60)

correct_vectorizer = CountVectorizer()
correct_vectorizer.fit(training_documents)

leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(training_documents + new_texts)

correct_vocabulary = set(correct_vectorizer.get_feature_names_out())
leaky_vocabulary = set(leaky_vectorizer.get_feature_names_out())

print("Training vocabulary:")
print(sorted(correct_vocabulary))

print()
print("Vocabulary after including test data:")
print(sorted(leaky_vocabulary))

print()
print("Words leaked from test data:")

for word in sorted(leaky_vocabulary - correct_vocabulary):
    print(word)


# ============================================================
# 12. NO MEANING
# ============================================================

print()
print("=" * 60)
print("12. NO MEANING")
print("=" * 60)

review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(reviews)

print("Vocabulary:")
print(review_vectorizer.get_feature_names_out().tolist())

print()
print("R1:")
print(review_matrix.toarray()[0])

print("R2:")
print(review_matrix.toarray()[1])

print("R3:")
print(review_matrix.toarray()[2])

similarity_matrix = cosine_similarity(review_matrix)

print()
print("Cosine similarity matrix:")
print(np.round(similarity_matrix, 2))

print()
print("R1 is not more similar to R2 than to R3.")
print("Both pairs share the same words: the, food, and was.")
print("CountVectorizer does not understand that tasty and delicious")
print("have similar meanings, or that terrible has an opposite meaning.")