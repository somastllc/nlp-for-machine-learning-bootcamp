import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

docs = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]
new_texts = ["Claim your free prize now", "Is the lunch meeting moved to tomorrow?"]
reviews = ["the food was tasty", "the food was delicious", "the food was terrible"]

def heading(title):
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)

# 1. One-hot encoding
heading("1) ONE-HOT ENCODING")
vocab = ["cash", "class", "free", "lunch", "notes", "win"]
def one_hot(word):
    vector = [0] * len(vocab)
    if word in vocab:
        vector[vocab.index(word)] = 1
    return vector
for word in ["free", "cash"]:
    print(word, "=", one_hot(word))
sentence = "win free cash".split()
print("Sentence:", sentence)
print("Vectors:", [one_hot(word) for word in sentence])
print("Number of vectors:", len(sentence))



# 2. Bag of Words
heading("2) BAG OF WORDS")
count_vec = CountVectorizer()
count_matrix = count_vec.fit_transform(docs)
terms = count_vec.get_feature_names_out()
counts = count_matrix.toarray()
print("Vocabulary:", terms.tolist())
print("Shape:", count_matrix.shape)
print("Document-term matrix:\n", counts)
for text in ["win free phone", "phone free win"]:
    print(text, "=", count_vec.transform([text]).toarray()[0].tolist())

# 3. Tokens and vocabulary

heading("3) TOKENS VS VOCABULARY")
analyzer = count_vec.build_analyzer()
token_lists = [analyzer(d) for d in docs]
dfs = Counter(word for tokens in token_lists for word in set(tokens))
print("Tokens per document:", [len(t) for t in token_lists])
print("Total tokens:", sum(map(len, token_lists)))
print("Vocabulary size:", len(terms))
print("Words in more than one document:", sorted(w for w, df in dfs.items() if df > 1))

# 4. Preprocessing

heading("4) PREPROCESSING")
for name, kwargs in [
    ("Default", {}),
    ("lowercase=False", {"lowercase": False}),
    ('stop_words="english"', {"stop_words": "english"})
]:
    vec = CountVectorizer(**kwargs).fit([docs[0]])
    print(name, ":", vec.get_feature_names_out().tolist())
stop_vec = CountVectorizer(stop_words="english")
stop_matrix = stop_vec.fit_transform(docs)
print("English stopwords on D1-D5:", stop_vec.get_feature_names_out().tolist())
print("'not' retained?", "not" in stop_vec.vocabulary_)

# 5. Vocabulary limits

heading("5) LIMITING THE VOCABULARY")
for name, kwargs in [("min_df=2", {"min_df": 2}), ("max_features=5", {"max_features": 5})]:
    vec = CountVectorizer(**kwargs).fit(docs)
    print(name, ":", vec.get_feature_names_out().tolist(),
          "size:", len(vec.get_feature_names_out()))
print("Full vocabulary size:", len(terms))

# 6. N-grams

heading("6) N-GRAMS")
for ngram_range in [(1, 1), (1, 2)]:
    vec = CountVectorizer(ngram_range=ngram_range).fit([docs[4]])
    print("D5", ngram_range, ":", vec.get_feature_names_out().tolist())
for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    vec = CountVectorizer(ngram_range=ngram_range).fit(docs)
    print("Corpus", ngram_range, "size:", len(vec.get_feature_names_out()))

# 7. TF-IDF

heading("7) TF-IDF")
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(docs)
tfidf_terms = tfidf_vec.get_feature_names_out()
tfidf = tfidf_matrix.toarray()
print("Vocabulary:", tfidf_terms.tolist())
print("TF-IDF matrix:\n", np.round(tfidf, 2))
for doc_index, doc_name in [(0, "D1"), (1, "D2")]:
    print(doc_name, "word | count | TF-IDF")
    for word in terms:
        count = counts[doc_index, list(terms).index(word)]
        if count:
            weight = tfidf[doc_index, list(tfidf_terms).index(word)]
            print(" ",word, "-> count:", int(count), "| TF-IDF:", round(weight, 3))

# 8. Textbook TF-IDF by hand for D2

heading("8) TF-IDF BY HAND")
N = len(docs)
total_d2 = int(counts[1].sum())
for word in ["cash", "today"]:
    j = list(terms).index(word)
    count = int(counts[1, j])
    df = int(np.count_nonzero(counts[:, j]))
    tf = count / total_d2
    idf = np.log(N / df)
    print(f"{word}: count={count}, total_words={total_d2}, TF={tf:.6f}, "
          f"df={df}, N={N}, IDF={idf:.6f}, TF-IDF={tf*idf:.6f}")
    print("scikit-learn normalized TF-IDF:",
          f"{tfidf[1, list(tfidf_terms).index(word)]:.6f}")

# 9. Sparse matrix

heading("9) SPARSE MATRIX")
total = counts.size
nonzero = int(np.count_nonzero(counts))
zeros = total - nonzero
print("Total cells:", total)
print("Non-zero cells:", nonzero)
print("Zero cells:", zeros)
print("Percentage of zeros:", f"{zeros / total * 100:.2f}%")

# 10. Fit and transform

heading("10) FIT AND TRANSFORM")
train_vec = CountVectorizer()
train_vec.fit(docs[:4])
train_vocab = set(train_vec.get_feature_names_out())
print("Training vocabulary:", sorted(train_vocab))
for text in new_texts:
    tokens = train_vec.build_analyzer()(text)
    print("Text:", text)
    print("Vector:", train_vec.transform([text]).toarray()[0].tolist())
    print("OOV:", [word for word in tokens if word not in train_vocab])

# 11. Data leakage

heading("11) DATA LEAKAGE")
train_only = CountVectorizer().fit(docs[:4])
train_plus_test = CountVectorizer().fit(docs[:4] + new_texts)
v1 = set(train_only.get_feature_names_out())
v2 = set(train_plus_test.get_feature_names_out())
print("Train-only vocabulary:", sorted(v1))
print("Train + test vocabulary:", sorted(v2))
print("Leaked words:", sorted(v2 - v1))

# 12. No meaning

heading("12) NO MEANING")
review_vec = CountVectorizer()
review_matrix = review_vec.fit_transform(reviews)
print("Vocabulary:", review_vec.get_feature_names_out().tolist())
print("Review vectors:\n", review_matrix.toarray())
similarities = cosine_similarity(review_matrix)
print("Cosine similarity:\n", np.round(similarities, 4))