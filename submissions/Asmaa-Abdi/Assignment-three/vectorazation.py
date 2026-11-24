import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer

np.set_printoptions(linewidth=200)

"""D1: Win a FREE phone now! Reply WIN to claim.
D2: Free entry: win cash prizes today. Call now.
D3: Are we still meeting for lunch today?
D4: Please send me the lecture notes before class.
D5: The lecture was not good, but the notes were great."""
#one_hot_encoding
one_hot_encoding = ["cash","class", "free", "lunch", "notes", "win"]
print("vocabulary", one_hot_encoding)

for word in "win free cash".split():
    one_hot = [1 if vocabulary_word == word else 0 for vocabulary_word in one_hot_encoding]
    print(word, "->", one_hot)

#bagofwords
documents = [
    "Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class.",
    "The lecture was not good, but the notes were great."
]

count_vector = CountVectorizer()
count_matrix = count_vector.fit_transform(documents)
print("documents", documents)
print("vocabulary", count_vector.get_feature_names_out().tolist())
print("(row) ,(columns): ", count_matrix.shape)
print(count_matrix.toarray())
print("win free phone:", count_vector.transform(["win free phone"]).toarray())
print("phone free win:", count_vector.transform(["phone free win"]).toarray())

#tokens and vocabulary size
all_tokens = " ".join(documents).split()
print("all tokens: ", len(all_tokens))
print("vocabulary size:" , len(count_vector.get_feature_names_out()))


#Preprocessing
message = ["Win a FREE phone now! Reply WIN to claim!"]
print("message:", message)
print("default:", CountVectorizer().fit(message).get_feature_names_out().tolist())
print("lowercasing:", CountVectorizer(lowercase=False).fit(message).get_feature_names_out().tolist())
print("stopword:", CountVectorizer(stop_words="english").fit(message).get_feature_names_out().tolist())


#limit
print("vocabulary size:", len(count_vector.get_feature_names_out()))
print("min_df:", CountVectorizer(min_df=2).fit(documents).get_feature_names_out().tolist())
print("max_features: ", CountVectorizer(max_features=5).fit(documents).get_feature_names_out().tolist())


#N-grams
n = ["The lecture was not good, but the notes were great."]
print("ngram:", n[0])
print("unigram:", CountVectorizer(ngram_range=(1, 1)).fit(n).get_feature_names_out().tolist())
print("unigram and bigram:", CountVectorizer(ngram_range=(1, 2)).fit(n).get_feature_names_out().tolist())
print("trigram , unigram and bigram:", CountVectorizer(ngram_range=(1, 3)).fit(documents).get_feature_names_out().tolist()) 


#TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)
vocabulary = count_vector.get_feature_names_out()
counts = count_matrix.toarray()[0]
weights = tfidf_matrix.toarray()[0]
print(tfidf_matrix.toarray().round(2))
print("count: ", documents[0])
for word, count, weight in zip(vocabulary, counts, weights):
    if count > 0:
        print(" ", word , "->count", count, "| tf-idf:", round(weight, 3))



#N1: Claim your free prize now
#N2: Is the lunch meeting moved to tomorrow?

#TF-IDF by hand
N = len(documents)
words = len(documents[0].split())
print("TF-IDF by hand:", documents[4], "===")
for word, df in [("cash", 1),  ("today", 2)]:
    tf = 1 / words
    idf = np.log(N / df)
    print(word, "| df=", df, "|tf=", tf, "| idf=", round(idf, 3), "| tf-idf=", round(tf * idf , 3))


#sparse matrix
total = count_matrix.shape[0] * count_matrix.shape[1]
print("total cells:", total)
print("non zero cells:", count_matrix.nnz)
print("zero cells:", total - count_matrix.nnz)


#Fit and transform
train = ["Win a FREE phone now! Reply WIN to claim.",
    "Free entry: win cash prizes today. Call now.",
    "Are we still meeting for lunch today?",
    "Please send me the lecture notes before class."]
vector = CountVectorizer()
x_train = vector.fit_transform(train)
print("vocabulary:", vector.get_feature_names_out().tolist())
print(x_train.toarray())
print("N1:", vector.transform(["Claim your free prize now"]).toarray())
print("N2:", vector.transform(["Is the lunch meeting moved to tomorrow?"]).toarray())

#leakage
leage = CountVectorizer()
leage.fit(train + ["Claim your free prize now", "Is the lunch meeting moved to tomorrow?"])
print("train only:", vector.get_feature_names_out().tolist())
print("train + test:", leage.get_feature_names_out().tolist())

#meaning
"""R1: the food was tasty
R2: the food was delicious
R3: the food was terrible"""

review = ["the food was tasty",
          "the food was delicious",
          "the food was terrible"
          ]
meaning = CountVectorizer()
meaning_matrix = meaning.fit_transform(review)
print("vocabulary:", meaning.get_feature_names_out().tolist())
print("tasty:", meaning_matrix.toarray()[0])
print("delicious:", meaning_matrix.toarray()[1])
print("terrible:", meaning_matrix.toarray()[2])
