"""Assignment 3 - Vectorization practical tasks.
Run: python vectorization_practice.py
Save output: python vectorization_practice.py > output.txt
Requirements: python -m pip install scikit-learn numpy
"""
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(linewidth=220, suppress=True)

# Assignment training corpus
D1 = "Win a FREE phone now! Reply WIN to claim."
D2 = "Free entry: win cash prizes today. Call now."
D3 = "Are we still meeting for lunch today?"
D4 = "Please send me the lecture notes before class."
D5 = "The lecture was not good, but the notes were great."
documents = [D1, D2, D3, D4, D5]
new_texts = [
    "Claim your free prize now",
    "Is the lunch meeting moved to tomorrow?"
]
reviews = [
    "the food was tasty",
    "the food was delicious",
    "the food was terrible"
]

def heading(number, title):
    print("\n" + "=" * 88)
    print(f"TASK {number}: {title}")
    print("=" * 88)

# 1. One-hot encoding
heading(1, "ONE-HOT ENCODING")
one_hot_vocab = ["cash", "class", "free", "lunch", "notes", "win"]
def one_hot(word):
    vector = [0] * len(one_hot_vocab)
    vector[one_hot_vocab.index(word)] = 1
    return vector
print("Vocabulary order:", one_hot_vocab)
for sentence in ["free cash", "win free cash"]:
    print(f"Sentence: {sentence!r}")
    for word in sentence.split():
        print(f"  {word:>6} -> {one_hot(word)}")
print("Vector length is fixed at 6; the number of word vectors depends on the number of words.")

# 2. Bag of Words / DTM
heading(2, "BAG OF WORDS AND DOCUMENT-TERM MATRIX")
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)
vocab = count_vectorizer.get_feature_names_out().tolist()
print("Vocabulary (alphabetical):", vocab)
print("Shape (documents, terms):", count_matrix.shape)
print("Column order:", vocab)
print("Full document-term matrix:")
print(count_matrix.toarray())
print("D1 count for 'win':", int(count_matrix.toarray()[0, vocab.index("win")]))
print("Reason: 'Win' and 'WIN' are both lowercased to 'win', so its count is 2.")
comparison_texts = ["win free phone", "phone free win"]
comparison_matrix = count_vectorizer.transform(comparison_texts)
print("Order test texts:", comparison_texts)
print("Order test vectors:")
print(comparison_matrix.toarray())
print("Vectors identical:", bool(np.array_equal(comparison_matrix.toarray()[0], comparison_matrix.toarray()[1])))

# 3. Tokens and vocabulary
heading(3, "TOKENS VERSUS VOCABULARY SIZE")
analyzer = count_vectorizer.build_analyzer()
token_lists = [analyzer(doc) for doc in documents]
all_tokens = [token for row in token_lists for token in row]
doc_freq = {word: sum(word in set(row) for row in token_lists) for word in vocab}
repeated_across_docs = {word: df for word, df in doc_freq.items() if df > 1}
print("Total tokens (default CountVectorizer analyzer):", len(all_tokens))
print("Vocabulary size:", len(vocab))
print("Words appearing in more than one document (document frequency):", repeated_across_docs)

# 4. Preprocessing inside CountVectorizer
heading(4, "PREPROCESSING INSIDE THE VECTORIZER")
for label, kwargs in [
    ("Default", {}),
    ("lowercase=False", {"lowercase": False}),
    ('stop_words="english"', {"stop_words": "english"})
]:
    vec = CountVectorizer(**kwargs)
    vec.fit([D1])
    print(f"{label} vocabulary:", vec.get_feature_names_out().tolist())
print("With default settings, punctuation is a separator, one-character tokens are excluded, and case is lowered.")
print("Therefore 'a' is excluded by the default token pattern; '!' is punctuation; 'Win' and 'WIN' merge as 'win'.")
default_d1 = set(CountVectorizer().fit([D1]).get_feature_names_out())
stop_d1 = set(CountVectorizer(stop_words="english").fit([D1]).get_feature_names_out())
print("Tokens removed by English stopword list from D1 vocabulary:", sorted(default_d1 - stop_d1))
stop_all = CountVectorizer(stop_words="english").fit(documents)
stop_all_vocab = set(stop_all.get_feature_names_out())
print("D1-D5 vocabulary with English stopwords:", sorted(stop_all_vocab))
print("Was 'not' removed?", "not" not in stop_all_vocab)
print("Meaning: removing 'not' removes an important negation cue from D5 and can change its sentiment meaning.")

# 5. Vocabulary limits
heading(5, "LIMITING THE VOCABULARY")
full_vec = CountVectorizer().fit(documents)
min_df_vec = CountVectorizer(min_df=2).fit(documents)
max_features_vec = CountVectorizer(max_features=5).fit(documents)
print("Full vocabulary size:", len(full_vec.get_feature_names_out()))
print("min_df=2 size:", len(min_df_vec.get_feature_names_out()))
print("min_df=2 vocabulary:", min_df_vec.get_feature_names_out().tolist())
print("max_features=5 size:", len(max_features_vec.get_feature_names_out()))
print("max_features=5 vocabulary:", max_features_vec.get_feature_names_out().tolist())
print("min_df keeps terms present in at least 2 documents; max_features keeps the top 5 by total term frequency.")

# 6. N-grams
heading(6, "N-GRAMS")
for ngram_range in [(1, 1), (1, 2)]:
    vec = CountVectorizer(ngram_range=ngram_range).fit([D5])
    print(f"D5 ngram_range={ngram_range}:", vec.get_feature_names_out().tolist())
print("Opinion-preserving bigram in D5: 'not good'.")
for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    vec = CountVectorizer(ngram_range=ngram_range).fit(documents)
    print(f"D1-D5 ngram_range={ngram_range}: vocabulary size = {len(vec.get_feature_names_out())}")

# 7. TF-IDF
heading(7, "TF-IDF")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tfidf_vocab = tfidf_vectorizer.get_feature_names_out().tolist()
print("Vocabulary:", tfidf_vocab)
print("TF-IDF matrix (rounded to 2 decimals):")
print(np.round(tfidf_matrix.toarray(), 2))
# Counts use the same fitted vocabulary, including the full corpus columns.
counts_for_tfidf_vocab = CountVectorizer(vocabulary=tfidf_vocab).fit_transform(documents).toarray()
weights = tfidf_matrix.toarray()
for row_index, label in [(0, "D1"), (1, "D2")]:
    print(f"{label} - term | count | TF-IDF weight")
    for term, count, weight in zip(tfidf_vocab, counts_for_tfidf_vocab[row_index], weights[row_index]):
        if count > 0:
            print(f"{term:>10} | {int(count):>5} | {weight:.6f}")
print("D2 has count 1 for each term; higher weights come from higher IDF for terms found in fewer documents.")
print("D1 'win' has count 2, which raises its term-frequency contribution and gives it the highest D1 weight.")

# 8. Textbook TF-IDF by hand for D2
heading(8, "TF-IDF BY HAND FOR D2")
N = len(documents)
d2_tokens = analyzer(D2)
d2_length = len(d2_tokens)
print("N documents:", N)
print("D2 tokens:", d2_tokens)
print("D2 token count:", d2_length)
hand_scores = {}
for term in ["cash", "today"]:
    count = d2_tokens.count(term)
    df = doc_freq[term]
    tf = count / d2_length
    idf = math.log(N / df)
    tfidf = tf * idf
    hand_scores[term] = tfidf
    print(f"{term}: count={count}; TF={count}/{d2_length}={tf:.6f}; df={df}; "
          f"IDF=ln({N}/{df})={idf:.6f}; TF-IDF={tf:.6f}*{idf:.6f}={tfidf:.6f}")
print("Hand ranking:", sorted(hand_scores, key=hand_scores.get, reverse=True))
print("The scikit-learn ranking in Task 7 is also cash above today, although its exact weights differ.")

# 9. Sparsity
heading(9, "SPARSE MATRIX")
rows, columns = count_matrix.shape
total_cells = rows * columns
nonzero_cells = int(count_matrix.nnz)
zero_cells = total_cells - nonzero_cells
zero_percentage = zero_cells / total_cells * 100
print("Matrix shape:", count_matrix.shape)
print("Total cells:", total_cells)
print("Non-zero cells:", nonzero_cells)
print("Zero cells:", zero_cells)
print(f"Percentage of zeros: {zero_percentage:.3f}%")

# 10. Fit and transform new texts
heading(10, "FIT ON TRAINING DATA AND TRANSFORM NEW TEXT")
train_docs = documents[:4]
train_vectorizer = CountVectorizer()
train_matrix = train_vectorizer.fit_transform(train_docs)
train_vocab = set(train_vectorizer.get_feature_names_out())
print("Training vocabulary (D1-D4):", sorted(train_vocab))
print("Training matrix:")
print(train_matrix.toarray())
for label, text in zip(["N1", "N2"], new_texts):
    vector = train_vectorizer.transform([text]).toarray()[0]
    tokens = analyzer(text)
    oov = sorted(set(tokens) - train_vocab)
    print(f"{label}: {text}")
    print("  Vector:", vector.tolist())
    print("  OOV words:", oov)
print("D2 has 'prizes' while N1 has 'prize'; default CountVectorizer does not reduce words to a common stem.")
print("Stemming or lemmatization from preprocessing can normalize plural/base forms, when appropriate.")

# 11. Data leakage
heading(11, "DATA LEAKAGE")
correct_vectorizer = CountVectorizer().fit(train_docs)
leaky_vectorizer = CountVectorizer().fit(train_docs + new_texts)
correct_vocab = set(correct_vectorizer.get_feature_names_out())
leaky_vocab = set(leaky_vectorizer.get_feature_names_out())
print("Vocabulary fitted on D1-D4 only:", sorted(correct_vocab))
print("Vocabulary incorrectly fitted on D1-D4 + N1 + N2:", sorted(leaky_vocab))
print("Words introduced from the test texts:", sorted(leaky_vocab - correct_vocab))
print("Correct order: split first; fit on training text; transform training and test text with the fitted vectorizer.")

# 12. No meaning: cosine similarity
heading(12, "NO MEANING: REVIEW VECTORS AND COSINE SIMILARITY")
review_vectorizer = CountVectorizer()
review_matrix = review_vectorizer.fit_transform(reviews)
print("Review vocabulary:", review_vectorizer.get_feature_names_out().tolist())
print("Review vectors:")
print(review_matrix.toarray())
similarity_matrix = cosine_similarity(review_matrix)
print("Cosine similarity matrix:")
print(np.round(similarity_matrix, 3))
print(f"Similarity R1 vs R2: {similarity_matrix[0, 1]:.3f}")
print(f"Similarity R1 vs R3: {similarity_matrix[0, 2]:.3f}")
print("R1 is not more similar to R2 than R3 under BoW cosine similarity; both share the same common words.")
print("The vectors do not encode the semantic relationship between 'tasty', 'delicious', and 'terrible'.")
