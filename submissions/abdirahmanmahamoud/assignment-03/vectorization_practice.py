import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

np.set_printoptions(linewidth=200)  # print each matrix row on one line

documents = [
    "win free cash now",
    "free prize claim now",
    "see you in class today",
    "please send the class notes",
]

# One-hot encoding of single words

one_hot_vocabulary = ["class", "free", "notes", "prize"]

print("Vocabulary:", one_hot_vocabulary)
for word in "free prize".split():
    one_hot = [1 if vocabulary_word == word else 0 for vocabulary_word in one_hot_vocabulary]
    print(word, "->", one_hot)

# Bag of Words and the document-term matrix

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

print("Documents: ", documents)
print("Vocabulary:", count_vectorizer.get_feature_names_out().tolist())
print("Shape (rows, columns):", count_matrix.shape)
print(count_matrix.toarray())
print("'free prize':", count_vectorizer.transform(["free prize"]).toarray())
print("'prize free':", count_vectorizer.transform(["prize free"]).toarray())

# Tokens vs vocabulary size

all_tokens = " ".join(documents).split()

print("Total tokens:   ", len(all_tokens))
print("Vocabulary size:", len(count_vectorizer.get_feature_names_out()))

# The vectorizer does its own preprocessing

message = ["FREE Prize! Call the office NOW"]

print("Message: ", message[0])
print("Default: ", CountVectorizer().fit(message).get_feature_names_out().tolist())
print("lowercase=False ", CountVectorizer(lowercase=False).fit(message).get_feature_names_out().tolist())
print("stop_words='english': ", CountVectorizer(stop_words="english").fit(message).get_feature_names_out().tolist())

# Limiting the vocabulary: min_df and max_features

print("Full vocabulary size:", len(count_vectorizer.get_feature_names_out()))
print("min_df=2:            ", CountVectorizer(min_df=2).fit(documents).get_feature_names_out().tolist())
print("max_features=5:      ", CountVectorizer(max_features=5).fit(documents).get_feature_names_out().tolist())

# N-grams

sentence = ["not a good movie"]

print("Sentence:", sentence[0])
print("Unigrams:", CountVectorizer(ngram_range=(1, 1)).fit(sentence).get_feature_names_out().tolist())
print("Unigrams and bigrams:", CountVectorizer(ngram_range=(1, 2)).fit(sentence).get_feature_names_out().tolist())

# TF-IDF

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

vocabulary = count_vectorizer.get_feature_names_out()
counts = count_matrix.toarray()[0]
weights = tfidf_matrix.toarray()[0]

print(tfidf_matrix.toarray().round(2))
print("Count vs TF-IDF for:", documents[0])
for word, count, weight in zip(vocabulary, counts, weights):
    if count > 0:
        print(" ", word, "-> count:", count, "| tf-idf:", round(weight, 3))

# TF-IDF by hand (textbook formula)
# TF = count / words in document, IDF = log(N / df), TF-IDF = TF x IDF.
N = len(documents)
words_in_document = len(documents[0].split())


for word, df in [("win", 1), ("free", 2)]:
    tf = 1 / words_in_document
    idf = np.log(N / df)
    print(word, "| df =", df, "| TF =", tf, "| IDF =", round(idf, 3), "| TF-IDF =", round(tf * idf, 3))

# Sparse matrices

total_cells = count_matrix.shape[0] * count_matrix.shape[1]

print("Total cells:   ", total_cells)
print("Non-zero cells:", count_matrix.nnz)
print("Zero cells:    ", total_cells - count_matrix.nnz)

# Fit on training text, transform new text

train_texts = ["free prize", "class notes"]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)

print("Vocabulary (from training text):", vectorizer.get_feature_names_out().tolist())
print("Train matrix:")
print(X_train.toarray())
print("'free notes':      ", vectorizer.transform(["free notes"]).toarray())
print("'free meeting':    ", vectorizer.transform(["free meeting"]).toarray())
print("'meeting tomorrow':", vectorizer.transform(["meeting tomorrow"]).toarray())

# Data leakage

leaky_vectorizer = CountVectorizer()
leaky_vectorizer.fit(train_texts + ["free meeting"])

print("Fitted on training text only:", vectorizer.get_feature_names_out().tolist())
print("Fitted on train + test text: ", leaky_vectorizer.get_feature_names_out().tolist())

# The main limitation: no meaning

reviews = ["the movie was good", "the movie was great"]
meaning_vectorizer = CountVectorizer()
meaning_matrix = meaning_vectorizer.fit_transform(reviews)

print("Vocabulary:", meaning_vectorizer.get_feature_names_out().tolist())
print("'the movie was good': ", meaning_matrix.toarray()[0])
print("'the movie was great':", meaning_matrix.toarray()[1])