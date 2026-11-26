# Assignment 3 - Vectorization practice

import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----- Data -----
docs = [
    "Win a FREE phone now! Reply WIN to claim.",            # D1
    "Free entry: win cash prizes today. Call now.",          # D2
    "Are we still meeting for lunch today?",                 # D3
    "Please send me the lecture notes before class.",        # D4
    "The lecture was not good, but the notes were great.",   # D5
]

new_texts = [
    "Claim your free prize now",                 # N1
    "Is the lunch meeting moved to tomorrow?",   # N2
]

reviews = [
    "the food was tasty",       # R1
    "the food was delicious",   # R2
    "the food was terrible",    # R3
]

print("===== TASK 1: One-hot encoding =====")
vocab = ["cash", "class", "free", "lunch", "notes", "win"]
print("Vocabulary:", vocab)

def one_hot(word):
    vector = [0] * len(vocab)        # start with all zeros
    index = vocab.index(word)        # find the word's position
    vector[index] = 1                # put a 1 there
    return vector

print("\n'free cash':")
for word in "free cash".split():
    print(word, one_hot(word))

print("\n'win free cash':")
for word in "win free cash".split():
    print(word, one_hot(word))

print("\n2 words -> 2 vectors, 3 words -> 3 vectors")


print("\n===== TASK 2: Bag of Words =====")
cv = CountVectorizer()
X = cv.fit_transform(docs)
words = cv.get_feature_names_out()

print("Vocabulary:", list(words))
print("Shape:", X.shape)
print("Document-term matrix:")
matrix = X.toarray()
for i in range(len(docs)):
    print("D" + str(i + 1), matrix[i].tolist())

print("\nCells with value 2:")
for i in range(len(docs)):
    for j in range(len(words)):
        if matrix[i][j] == 2:
            print("D" + str(i + 1), words[j], "= 2")

a = cv.transform(["win free phone"]).toarray()[0]
b = cv.transform(["phone free win"]).toarray()[0]
print("\n'win free phone':", a.tolist())
print("'phone free win':", b.tolist())
print("Same?", a.tolist() == b.tolist())


print("\n===== TASK 3: Tokens vs vocabulary size =====")
print("Total tokens:", matrix.sum())
print("Vocabulary size:", len(words))

print("Words in more than one document:")
for j in range(len(words)):
    doc_count = 0
    for i in range(len(docs)):
        if matrix[i][j] > 0:
            doc_count = doc_count + 1
    if doc_count > 1:
        print(" ", words[j], "-> in", doc_count, "documents")


print("\n===== TASK 4: Preprocessing (fit on D1 only) =====")
d1 = [docs[0]]

v1 = CountVectorizer().fit(d1)
print("Default:          ", list(v1.get_feature_names_out()))

v2 = CountVectorizer(lowercase=False).fit(d1)
print("lowercase=False:  ", list(v2.get_feature_names_out()))

v3 = CountVectorizer(stop_words="english").fit(d1)
print("stop_words:       ", list(v3.get_feature_names_out()))

removed = set(v1.get_feature_names_out()) - set(v3.get_feature_names_out())
print("Removed by stopwords:", sorted(removed))

v4 = CountVectorizer(stop_words="english").fit(docs)
print("\nstop_words on D1-D5:", list(v4.get_feature_names_out()))
print("Is 'not' still there?", "not" in v4.get_feature_names_out())


print("\n===== TASK 5: Limiting the vocabulary =====")
v_min = CountVectorizer(min_df=2).fit(docs)
v_max = CountVectorizer(max_features=5).fit(docs)
print("Full vocabulary size:", len(words))
print("min_df=2:       size", len(v_min.vocabulary_), list(v_min.get_feature_names_out()))
print("max_features=5: size", len(v_max.vocabulary_), list(v_max.get_feature_names_out()))


print("\n===== TASK 6: N-grams =====")
d5 = [docs[4]]
uni = CountVectorizer(ngram_range=(1, 1)).fit(d5)
bi = CountVectorizer(ngram_range=(1, 2)).fit(d5)
print("D5 (1,1):", list(uni.get_feature_names_out()))
print("D5 (1,2):", list(bi.get_feature_names_out()))

print("\nVocabulary size on D1-D5:")
for n in [1, 2, 3]:
    v = CountVectorizer(ngram_range=(1, n)).fit(docs)
    print("  ngram_range=(1," + str(n) + "):", len(v.vocabulary_))


print("\n===== TASK 7: TF-IDF =====")
tfidf = TfidfVectorizer()
T = tfidf.fit_transform(docs).toarray()

print("Vocabulary:", list(words))
for i in range(len(docs)):
    print("D" + str(i + 1), T[i].round(2).tolist())

for i in [0, 1]:   # D1 and D2
    print("\nD" + str(i + 1) + ": word, count, tf-idf")
    for j in range(len(words)):
        if matrix[i][j] > 0:
            print(" ", words[j], matrix[i][j], round(T[i][j], 2))


print("\n===== TASK 8: TF-IDF by hand (D2) =====")
N = 5                       # number of documents
words_in_d2 = 8             # free entry win cash prizes today call now

# cash: count 1 in D2, appears only in D2 (df = 1)
tf_cash = 1 / words_in_d2
idf_cash = math.log(N / 1)
print("cash : TF =", round(tf_cash, 4), " IDF =", round(idf_cash, 4),
      " TF-IDF =", round(tf_cash * idf_cash, 4))

# today: count 1 in D2, appears in D2 and D3 (df = 2)
tf_today = 1 / words_in_d2
idf_today = math.log(N / 2)
print("today: TF =", round(tf_today, 4), " IDF =", round(idf_today, 4),
      " TF-IDF =", round(tf_today * idf_today, 4))

cash_col = list(words).index("cash")
today_col = list(words).index("today")
print("scikit-learn: cash =", round(T[1][cash_col], 4),
      " today =", round(T[1][today_col], 4))


print("\n===== TASK 9: Sparse matrix =====")
total = X.shape[0] * X.shape[1]
non_zero = X.nnz
zeros = total - non_zero
print("Total cells:", total)
print("Non-zero cells:", non_zero)
print("Zero cells:", zeros)
print("Percentage of zeros:", round(zeros / total * 100, 2), "%")


print("\n===== TASK 10: Fit on D1-D4, transform N1 and N2 =====")
train = docs[:4]
cv_train = CountVectorizer().fit(train)
train_words = cv_train.get_feature_names_out()
print("Training vocabulary:", list(train_words))

new_X = cv_train.transform(new_texts).toarray()
print("N1:", new_X[0].tolist())
print("N2:", new_X[1].tolist())

tokenize = cv_train.build_analyzer()   # splits text the same way the vectorizer does
for text in new_texts:
    oov = [w for w in tokenize(text) if w not in train_words]
    print("OOV words in '" + text + "':", oov)


print("\n===== TASK 11: Data leakage =====")
clean = CountVectorizer().fit(train)
leaky = CountVectorizer().fit(train + new_texts)
clean_words = set(clean.get_feature_names_out())
leaky_words = set(leaky.get_feature_names_out())
print("Train only:  ", len(clean_words), "words")
print("Train + test:", len(leaky_words), "words")
print("Leaked words:", sorted(leaky_words - clean_words))

print("\n===== TASK 12: No meaning =====")
cv_rev = CountVectorizer()
R = cv_rev.fit_transform(reviews)
print("Vocabulary:", list(cv_rev.get_feature_names_out()))
R_array = R.toarray()
for i in range(3):
    print("R" + str(i + 1), R_array[i].tolist())

sim = cosine_similarity(R)
print("\nCosine similarity:")
for i in range(3):
    print("R" + str(i + 1), sim[i].round(2).tolist())
print("R1 vs R2 (tasty vs delicious):", round(sim[0][1], 2))
print("R1 vs R3 (tasty vs terrible): ", round(sim[0][2], 2))