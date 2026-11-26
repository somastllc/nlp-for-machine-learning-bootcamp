# Assignment 3 — Vectorization (practice script)
# Follows the same steps as code/vectorization.py.
# Run: python vectorization_practice.py > output.txt

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=200)  # print each matrix row on one line

documents = [
    "Win a FREE phone now! Reply WIN to claim.",             # D1
    "Free entry: win cash prizes today. Call now.",          # D2
    "Are we still meeting for lunch today?",                 # D3
    "Please send me the lecture notes before class.",        # D4
    "The lecture was not good, but the notes were great.",   # D5
]

new_texts = [
    "Claim your free prize now",                             # N1
    "Is the lunch meeting moved to tomorrow?",               # N2
]

# --------------------------------
# 1) One-hot encoding of single words
# Each vocabulary word gets a vector with a single 1 in its own position.
# --------------------------------
one_hot_vocabulary = ["cash", "class", "free", "lunch", "notes", "win"]

print("\n=== 1) ONE-HOT ENCODING ===")
print("Vocabulary:", one_hot_vocabulary)
print("Sentence: free cash")
for word in "free cash".split():
    one_hot = [1 if vocabulary_word == word else 0 for vocabulary_word in one_hot_vocabulary]
    print(word, "->", one_hot)
print("Sentence: win free cash")
for word in "win free cash".split():
    one_hot = [1 if vocabulary_word == word else 0 for vocabulary_word in one_hot_vocabulary]
    print(word, "->", one_hot)
print("'free cash' gives", len("free cash".split()), "vectors, 'win free cash' gives", len("win free cash".split()), "vectors.")
print("One vector per word, so the number of vectors depends on the sentence length.")

# --------------------------------
# 2) Bag of Words and the document-term matrix
# fit learns the vocabulary; each word becomes one column, each document one row.
# --------------------------------
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print("\n=== 2) BAG OF WORDS ===")
print("Vocabulary:", count_vectorizer.get_feature_names_out().tolist())
print("Shape (rows, columns):", count_matrix.shape)
print(count_matrix.toarray())

vocabulary = count_vectorizer.get_feature_names_out()
counts = count_matrix.toarray()[0]
for word, count in zip(vocabulary, counts):
    if count == 2:
        print("Cell with value 2 in D1:", word, "(it appears twice: 'Win' and 'WIN', lowercased to one word)")

print("'win free phone':", count_vectorizer.transform(["win free phone"]).toarray())
print("'phone free win':", count_vectorizer.transform(["phone free win"]).toarray())
print("Same vector?", np.array_equal(count_vectorizer.transform(["win free phone"]).toarray(),
                                     count_vectorizer.transform(["phone free win"]).toarray()))

# --------------------------------
# 3) Tokens vs vocabulary size
# Tokens count every word occurrence; vocabulary size counts unique words (columns).
# --------------------------------
analyzer = count_vectorizer.build_analyzer()  # same tokenizer the vectorizer uses
all_tokens = [token for document in documents for token in analyzer(document)]
document_frequency = (count_matrix.toarray() > 0).sum(axis=0)

print("\n=== 3) TOKENS vs VOCABULARY SIZE ===")
print("Total tokens (counted like the vectorizer):", len(all_tokens))
print("Check: sum of all cells in the count matrix:", int(count_matrix.sum()))
print("Vocabulary size:", len(vocabulary))
print("Words in more than one document:", [word for word, df in zip(vocabulary, document_frequency) if df > 1])

# --------------------------------
# 4) The vectorizer does its own preprocessing
# By default CountVectorizer lowercases, drops punctuation and one-letter words.
# stop_words="english" removes common words — check what it removes.
# --------------------------------
message = [documents[0]]
default_words = CountVectorizer().fit(message).get_feature_names_out().tolist()
stopword_words = CountVectorizer(stop_words="english").fit(message).get_feature_names_out().tolist()

print("\n=== 4) PREPROCESSING INSIDE THE VECTORIZER ===")
print("Message:              ", message[0])
print("Default:              ", default_words)
print("lowercase=False:      ", CountVectorizer(lowercase=False).fit(message).get_feature_names_out().tolist())
print("stop_words='english': ", stopword_words)
print("Removed by the stopword list:", [word for word in default_words if word not in stopword_words])
all_stopword_words = CountVectorizer(stop_words="english").fit(documents).get_feature_names_out().tolist()
print("stop_words='english' on D1-D5:", all_stopword_words)
print("Was 'not' removed?", "not" not in all_stopword_words)

# --------------------------------
# 5) Limiting the vocabulary: min_df and max_features
# min_df=2 keeps words found in at least 2 documents.
# max_features=5 keeps only the 5 most frequent words
# (when several words have the same count, scikit-learn does not guarantee which ones are kept).
# --------------------------------
print("\n=== 5) LIMITING THE VOCABULARY ===")
print("Full vocabulary size:", len(vocabulary))
print("min_df=2:            ", CountVectorizer(min_df=2).fit(documents).get_feature_names_out().tolist())
print("min_df=2 size:       ", len(CountVectorizer(min_df=2).fit(documents).get_feature_names_out()))
print("max_features=5:      ", CountVectorizer(max_features=5).fit(documents).get_feature_names_out().tolist())
print("max_features=5 size: ", len(CountVectorizer(max_features=5).fit(documents).get_feature_names_out()))

# --------------------------------
# 6) N-grams
# ngram_range=(1, 2) adds bigrams (two neighboring words) to the single words.
# --------------------------------
sentence = [documents[4]]

print("\n=== 6) N-GRAMS ===")
print("Sentence:            ", sentence[0])
print("Unigrams:            ", CountVectorizer(ngram_range=(1, 1)).fit(sentence).get_feature_names_out().tolist())
print("Unigrams and bigrams:", CountVectorizer(ngram_range=(1, 2)).fit(sentence).get_feature_names_out().tolist())
print("Vocabulary size on D1-D5:")
print("(1, 1):", len(CountVectorizer(ngram_range=(1, 1)).fit(documents).get_feature_names_out()))
print("(1, 2):", len(CountVectorizer(ngram_range=(1, 2)).fit(documents).get_feature_names_out()))
print("(1, 3):", len(CountVectorizer(ngram_range=(1, 3)).fit(documents).get_feature_names_out()))

# --------------------------------
# 7) TF-IDF
# Same vocabulary and shape as counts, but each cell is a weight.
# Words found in fewer documents get higher weights. Weights are not probabilities.
# --------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\n=== 7) TF-IDF ===")
print(tfidf_matrix.toarray().round(2))
for row in [0, 1]:
    counts = count_matrix.toarray()[row]
    weights = tfidf_matrix.toarray()[row]
    print("Count vs TF-IDF for:", documents[row])
    for word, count, weight in zip(vocabulary, counts, weights):
        if count > 0:
            print(" ", word, "-> count:", count, "| tf-idf:", round(weight, 3))

# --------------------------------
# 8) TF-IDF by hand (textbook formula) for D2
# TF = count / words in document, IDF = log(N / df), TF-IDF = TF x IDF.
# --------------------------------
N = len(documents)
words_in_document = len(count_vectorizer.build_analyzer()(documents[1]))
hand_tfidf = []

print("\n=== 8) TF-IDF BY HAND FOR:", documents[1], "===")
print("N =", N, "| words in D2 =", words_in_document)
for word in ["cash", "today"]:
    df = int(document_frequency[vocabulary.tolist().index(word)])
    tf = 1 / words_in_document
    idf = np.log(N / df)
    hand_tfidf.append(tf * idf)
    print(word, "| df =", df, "| TF =", tf, "| IDF =", round(idf, 3), "| TF-IDF =", round(tf * idf, 3))
sklearn_cash = tfidf_matrix.toarray()[1][vocabulary.tolist().index("cash")]
sklearn_today = tfidf_matrix.toarray()[1][vocabulary.tolist().index("today")]
print("scikit-learn: cash =", round(sklearn_cash, 3), "| today =", round(sklearn_today, 3))
print("Same ranking (cash higher than today)?", (hand_tfidf[0] > hand_tfidf[1]) == (sklearn_cash > sklearn_today))

# --------------------------------
# 9) Sparse matrices
# Most cells are zero, so scikit-learn stores only the non-zero cells.
# --------------------------------
total_cells = count_matrix.shape[0] * count_matrix.shape[1]

print("\n=== 9) SPARSE MATRIX ===")
print("Total cells:   ", total_cells)
print("Non-zero cells:", count_matrix.nnz)
print("Zero cells:    ", total_cells - count_matrix.nnz)
print("Percentage of zeros:", round(100 * (total_cells - count_matrix.nnz) / total_cells, 1), "%")

# --------------------------------
# 10) Fit on training text, transform new text
# fit_transform learns the vocabulary from training text only.
# transform reuses it: new (out-of-vocabulary) words have no column and are ignored.
# --------------------------------
train_texts = documents[:4]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)

print("\n=== 10) FIT ON TRAIN, TRANSFORM NEW TEXT ===")
print("Vocabulary (from training text):", vectorizer.get_feature_names_out().tolist())
print("Train matrix shape:", X_train.shape)
for text in new_texts:
    print(text)
    print("  vector:", vectorizer.transform([text]).toarray())
    print("  out-of-vocabulary words:", [word for word in vectorizer.build_analyzer()(text)
                                          if word not in vectorizer.get_feature_names_out()])
print("'prize' is unknown because D2 contains 'prizes', a different word form.")
print("Stemming or lemmatization (Assignment 2) would turn 'prizes' into 'prize'.")

# --------------------------------
# 11) Data leakage
# Fitting on ALL text lets the test words into the vocabulary before evaluation.
# --------------------------------
leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(train_texts + new_texts)

print("\n=== 11) DATA LEAKAGE ===")
print("Fitted on training text only:", vectorizer.get_feature_names_out().tolist())
print("Fitted on train + test text: ", leaky_vectorizer.get_feature_names_out().tolist())
print("Leaked words:", [word for word in leaky_vectorizer.get_feature_names_out()
                        if word not in vectorizer.get_feature_names_out()])

# --------------------------------
# 12) The main limitation: no meaning
# "tasty" and "delicious" mean almost the same, but they are separate, unrelated columns.
# --------------------------------
reviews = ["the food was tasty", "the food was delicious", "the food was terrible"]
meaning_vectorizer = CountVectorizer()
meaning_matrix = meaning_vectorizer.fit_transform(reviews)
similarity = cosine_similarity(meaning_matrix)

print("\n=== 12) NO MEANING: 'tasty' vs 'delicious' vs 'terrible' ===")
print("Vocabulary:", meaning_vectorizer.get_feature_names_out().tolist())
print("R1 'the food was tasty':    ", meaning_matrix.toarray()[0])
print("R2 'the food was delicious':", meaning_matrix.toarray()[1])
print("R3 'the food was terrible': ", meaning_matrix.toarray()[2])
print("Cosine similarity matrix:")
print(similarity.round(2))
print("R1-R2:", round(similarity[0][1], 2), "| R1-R3:", round(similarity[0][2], 2))
print("R1 more similar to R2 than to R3?", similarity[0][1] > similarity[0][2])
