import math

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = {
    "D1": "Win a FREE phone now! Reply WIN to claim.",
    "D2": "Free entry: win cash prizes today. Call now.",
    "D3": "Are we still meeting for lunch today?",
    "D4": "Please send me the lecture notes before class.",
    "D5": "The lecture was not good, but the notes were great.",
}
NEW = {
    "N1": "Claim your free prize now",
    "N2": "Is the lunch meeting moved to tomorrow?",
}
REVIEWS = {
    "R1": "the food was tasty",
    "R2": "the food was delicious",
    "R3": "the food was terrible",
}

D = list(DOCS.values())
D_NAMES = list(DOCS.keys())
NEW_TEXTS = list(NEW.values())


def heading(number, title):
    print("\n=== " + str(number) + ") " + title + " ===")


def show_matrix(row_names, matrix, col_names, decimals=None):
    width = max(max(len(name) for name in col_names), 6) + 2
    print("".ljust(6) + "".join(name.ljust(width) for name in col_names))
    for row_name, row in zip(row_names, matrix):
        if decimals is None:
            cells = [str(int(value)) for value in row]
        else:
            cells = [f"{value:.{decimals}f}" for value in row]
        print(row_name.ljust(6) + "".join(cell.ljust(width) for cell in cells))


def ordered_vocabulary(vectorizer):
    pairs = {word: int(index) for word, index in vectorizer.vocabulary_.items()}
    return dict(sorted(pairs.items(), key=lambda pair: pair[1]))


heading(1, "ONE HOT ENCODING")
one_hot_vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]


def one_hot(word):
    return [1 if word == entry else 0 for entry in one_hot_vocabulary]


print("Vocabulary:", one_hot_vocabulary)
for sentence in ["free cash", "win free cash"]:
    words = sentence.split()
    vectors = [one_hot(word) for word in words]
    print("\nSentence:", sentence)
    for word, vector in zip(words, vectors):
        print(word.ljust(6), vector)
    print("Number of vectors:", len(vectors), "| length of each vector:", len(vectors[0]))

heading(2, "BAG OF WORDS")
count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(D)
features = list(count_vectorizer.get_feature_names_out())
print("Vocabulary (word: column):", ordered_vocabulary(count_vectorizer))
print("Shape:", X.shape)
print("\nDocument term matrix:")
show_matrix(D_NAMES, X.toarray(), features)

counts = X.toarray()
print("\nCells with a value of 2 or more:")
for row_index, name in enumerate(D_NAMES):
    for column_index, word in enumerate(features):
        if counts[row_index][column_index] >= 2:
            print("   ", name, word, counts[row_index][column_index])

vector_a = count_vectorizer.transform(["win free phone"]).toarray()[0]
vector_b = count_vectorizer.transform(["phone free win"]).toarray()[0]
print("\nColumns:          ", features)
print("win free phone:   ", vector_a.tolist())
print("phone free win:   ", vector_b.tolist())
print("Same vector:", np.array_equal(vector_a, vector_b))

heading(3, "TOKENS AGAINST VOCABULARY SIZE")
analyzer = count_vectorizer.build_analyzer()
total_tokens = 0
for name, text in DOCS.items():
    tokens = analyzer(text)
    total_tokens += len(tokens)
    print(name, len(tokens), tokens)
print("\nTotal tokens (word occurrences):", total_tokens)
print("Vocabulary size:", len(features))
document_frequency = (counts > 0).sum(axis=0)
print("Words that appear in more than one document (word, number of documents):")
print([(word, int(df)) for word, df in zip(features, document_frequency) if df > 1])

heading(4, "PREPROCESSING INSIDE THE VECTORIZER")
d1_only = [DOCS["D1"]]
default_vectorizer = CountVectorizer().fit(d1_only)
case_vectorizer = CountVectorizer(lowercase=False).fit(d1_only)
stop_vectorizer = CountVectorizer(stop_words="english").fit(d1_only)
default_words = list(default_vectorizer.get_feature_names_out())
case_words = list(case_vectorizer.get_feature_names_out())
stop_words_d1 = list(stop_vectorizer.get_feature_names_out())
print("D1 text:", DOCS["D1"])
print("Default:          ", default_words, len(default_words))
print("lowercase=False:  ", case_words, len(case_words))
print("stop_words english:", stop_words_d1, len(stop_words_d1))
print("Removed by the stopword list:", sorted(set(default_words) - set(stop_words_d1)))

stop_all = CountVectorizer(stop_words="english").fit(D)
stop_all_words = list(stop_all.get_feature_names_out())
print("\nstop_words english on D1 to D5:", stop_all_words, len(stop_all_words))
print("not is in the english stopword list:", "not" in ENGLISH_STOP_WORDS)
print("not is in the vocabulary:", "not" in stop_all_words)
print("D5 tokens, default:        ", analyzer(DOCS["D5"]))
print("D5 tokens, stop_words english:", stop_all.build_analyzer()(DOCS["D5"]))

heading(5, "LIMITING THE VOCABULARY")
min_df_vectorizer = CountVectorizer(min_df=2).fit(D)
max_features_vectorizer = CountVectorizer(max_features=5).fit(D)
min_df_words = list(min_df_vectorizer.get_feature_names_out())
max_features_words = list(max_features_vectorizer.get_feature_names_out())
total_counts = counts.sum(axis=0)
print("Full vocabulary size:", len(features))
print("\nmin_df=2:", min_df_words, "size", len(min_df_words))
print("max_features=5:", max_features_words, "size", len(max_features_words))
print("\nTotal count and number of documents for each word in the full vocabulary:")
for word, total, df in zip(features, total_counts, document_frequency):
    print("   ", word.ljust(10), "count", int(total), "documents", int(df))

heading(6, "N GRAMS")
d5_only = [DOCS["D5"]]
unigram_d5 = CountVectorizer(ngram_range=(1, 1)).fit(d5_only)
bigram_d5 = CountVectorizer(ngram_range=(1, 2)).fit(d5_only)
print("D5 text:", DOCS["D5"])
print("ngram_range=(1, 1):", list(unigram_d5.get_feature_names_out()), len(unigram_d5.vocabulary_))
print("ngram_range=(1, 2):", list(bigram_d5.get_feature_names_out()), len(bigram_d5.vocabulary_))
print("Bigrams that contain the word not:",
      [term for term in bigram_d5.get_feature_names_out() if " " in term and "not" in term.split()])
print("\nVocabulary size on D1 to D5:")
sizes = {}
for n in [1, 2, 3]:
    size = len(CountVectorizer(ngram_range=(1, n)).fit(D).vocabulary_)
    sizes[n] = size
    print("   ngram_range=(1, " + str(n) + "):", size)
print("Size with bigrams compared to unigrams:", round(sizes[2] / sizes[1], 2), "times")
print("Size with trigrams compared to unigrams:", round(sizes[3] / sizes[1], 2), "times")

heading(7, "TF IDF")
tfidf = TfidfVectorizer()
T = tfidf.fit_transform(D)
tfidf_features = list(tfidf.get_feature_names_out())
assert tfidf_features == features
print("TF IDF matrix, rounded to 2 decimals:")
show_matrix(D_NAMES, T.toarray(), tfidf_features, decimals=2)
for name in ["D1", "D2"]:
    row_index = D_NAMES.index(name)
    rows = []
    for column_index, word in enumerate(features):
        if counts[row_index][column_index] > 0:
            rows.append((word, int(counts[row_index][column_index]),
                         int(document_frequency[column_index]),
                         tfidf.idf_[column_index], T[row_index, column_index]))
    rows.sort(key=lambda item: item[4], reverse=True)
    print("\n" + name + ": " + DOCS[name])
    print("word".ljust(10), "count".ljust(7), "df".ljust(5), "idf".ljust(7), "tfidf")
    for word, count, df, idf, weight in rows:
        print(word.ljust(10), str(count).ljust(7), str(df).ljust(5), f"{idf:.2f}".ljust(7), f"{weight:.2f}")

heading(8, "TF IDF BY HAND FOR D2")
d2_tokens = analyzer(DOCS["D2"])
number_of_documents = len(D)
print("D2 tokens:", d2_tokens)
print("Number of words in D2:", len(d2_tokens))
print("Number of documents N:", number_of_documents)
print("Formula: tf = count / words in D2, idf = log(N / df) with the natural log, tfidf = tf * idf")
d2_row = D_NAMES.index("D2")
hand_values = {}
sklearn_values = {}
for word in ["cash", "today"]:
    count = d2_tokens.count(word)
    tf = count / len(d2_tokens)
    column_index = features.index(word)
    df = int(document_frequency[column_index])
    idf = math.log(number_of_documents / df)
    value = tf * idf
    hand_values[word] = value
    sklearn_values[word] = T[d2_row, column_index]
    print("\nWord:", word)
    print("   count:", count)
    print("   tf = " + str(count) + " / " + str(len(d2_tokens)) + " =", round(tf, 4))
    print("   df:", df)
    print("   N / df =", number_of_documents, "/", df, "=", round(number_of_documents / df, 4))
    print("   idf = log(N / df) =", round(idf, 4))
    print("   tfidf = tf * idf =", round(value, 4))
    print("   scikit learn idf (smoothed):", round(float(tfidf.idf_[column_index]), 4))
    print("   scikit learn tfidf weight in D2:", round(float(sklearn_values[word]), 4))
hand_rank = sorted(hand_values, key=hand_values.get, reverse=True)
sklearn_rank = sorted(sklearn_values, key=sklearn_values.get, reverse=True)
print("\nRanking by hand:        ", hand_rank)
print("Ranking by scikit learn:", sklearn_rank)
print("Same ranking:", hand_rank == sklearn_rank)

heading(9, "SPARSE MATRIX")
total_cells = X.shape[0] * X.shape[1]
non_zero = X.nnz
zero_cells = total_cells - non_zero
print("Type:", type(X).__name__)
print("Shape:", X.shape)
print("Total cells:", total_cells)
print("Non zero cells:", non_zero)
print("Zero cells:", zero_cells)
print("Percentage of zeros:", round(100 * zero_cells / total_cells, 2))

heading(10, "FIT AND TRANSFORM")
train_texts = D[:4]
fit_vectorizer = CountVectorizer().fit(train_texts)
train_features = list(fit_vectorizer.get_feature_names_out())
new_vectors = fit_vectorizer.transform(NEW_TEXTS).toarray()
print("Fitted on D1 to D4. Vocabulary:", train_features, len(train_features))
print("\nVectors for the new texts:")
show_matrix(list(NEW.keys()), new_vectors, train_features)
fit_analyzer = fit_vectorizer.build_analyzer()
print()
for name, text in NEW.items():
    tokens = fit_analyzer(text)
    unknown = [token for token in tokens if token not in fit_vectorizer.vocabulary_]
    print(name, "tokens:", tokens)
    print(name, "unknown words:", unknown)
stemmer = PorterStemmer()
print("\nprizes in the vocabulary:", "prizes" in fit_vectorizer.vocabulary_)
print("prize in the vocabulary: ", "prize" in fit_vectorizer.vocabulary_)
print("Porter stem of prizes:", stemmer.stem("prizes"))
print("Porter stem of prize: ", stemmer.stem("prize"))

heading(11, "DATA LEAKAGE")
clean_vectorizer = CountVectorizer().fit(D[:4])
leaky_vectorizer = CountVectorizer().fit(D[:4] + NEW_TEXTS)
clean_words = list(clean_vectorizer.get_feature_names_out())
leaky_words = list(leaky_vectorizer.get_feature_names_out())
leaked = sorted(set(leaky_words) - set(clean_words))
print("Fitted on D1 to D4 only:", clean_words, len(clean_words))
print("Fitted on D1 to D4 and N1 and N2:", leaky_words, len(leaky_words))
print("Words that leaked from the test texts:", leaked, len(leaked))

heading(12, "NO MEANING")
review_names = list(REVIEWS.keys())
review_vectorizer = CountVectorizer()
R = review_vectorizer.fit_transform(list(REVIEWS.values()))
review_features = list(review_vectorizer.get_feature_names_out())
print("Vocabulary:", review_features)
print("Vectors:")
show_matrix(review_names, R.toarray(), review_features)
similarity = cosine_similarity(R)
print("\nCosine similarity:")
show_matrix(review_names, similarity, review_names, decimals=2)
print("\nR1 and R2 (tasty, delicious):", round(float(similarity[0][1]), 2))
print("R1 and R3 (tasty, terrible): ", round(float(similarity[0][2]), 2))
print("R1 is more similar to R2 than to R3:", bool(similarity[0][1] > similarity[0][2]))
