# Assignment 3: Vectorization

## Part A: Research

### 1. Text vectorization, fixed length features and one hot encoding

Text vectorization means turning a piece of text into a list of numbers, called a vector, so that a Machine Learning model can work with it. A model cannot do arithmetic on the word "free", but it can do arithmetic on a number.

Most models need every document to have the same number of numerical features because a model is a fixed formula. Think of a table where each row is one example and each column is one feature. A linear model learns one weight for each column, and a neural network has an input layer with a fixed number of slots. A short message and a long email must both fit that same table. Vectorizers solve this by giving each word in the vocabulary one column. The scikit learn guide says each term found during fitting gets "a unique integer index corresponding to a column in the resulting matrix" [3]. Jurafsky and Martin say the same thing in other words: the document vectors have the size of the vocabulary, written |V| [2].

**One hot encoding of words.** One hot encoding gives each word a vector as long as the vocabulary. It has a single 1 at the position of that word and 0 everywhere else. Jurafsky and Martin describe the starting point as a word being nothing more than "a string of letters, or an index in a vocabulary list" [2].

**Why it is not enough for a whole document.** There are four problems.

1. A document has many words, so it gets many vectors. A 3 word message gets 3 vectors and a 12 word message gets 12. The model needs one input of one fixed size.
2. The vectors describe single words, not the document. Nothing in them says which words belong together.
3. Every pair of words is equally far apart. The vectors for "tea" and "coffee" are no closer than the vectors for "tea" and "invoice".
4. The vectors are huge and nearly all zeros. With 50,000 words in the vocabulary, each word needs 50,000 numbers.

**A real life example.** Think of a phone company that wants to block scam text messages. Its vocabulary has 50,000 words. The message "win free cash" becomes three one hot vectors of 50,000 numbers each. The message "call me" becomes two. The spam filter cannot take inputs that change size. If the company adds the three one hot vectors together, it gets one vector of 50,000 numbers that counts each word. That sum is the Bag of Words vector, which I explain in the next question. Part B, task 1 shows the same thing at a small scale.

### 2. Bag of Words and the document term matrix

Bag of Words describes a document only by which words it contains and how many times each appears. Manning, Raghavan and Schütze write that "the exact ordering of the terms in a document is ignored but the number of occurrences of each term is material" [1].

**The document term matrix.** In scikit learn it is a table where:

* each row is one document,
* each column is one word of the vocabulary,
* each cell is the number of times that word appears in that document [3].

A note on names. Jurafsky and Martin draw the same table the other way round, with a row for each word, and call it a term document matrix [2]. It is the same information turned on its side.

**Why it is called a bag.** Picture a bag of marbles. You can see which colors are inside and how many of each, but the bag keeps no order. The scikit learn guide says documents are described by word occurrences "while completely ignoring the relative position information of the words" [3]. The IR book gives a classic example: "Mary is quicker than John" and "John is quicker than Mary" get the same vector, although the meaning is the opposite [1].

**A task where losing order does little harm.** Spam detection and topic sorting. A message with "win", "free" and "cash" looks suspicious wherever those words stand. A search engine also works well by adding up the weights of the query words found in a document [1].

**A task where it causes real problems.** Sentiment analysis with negation. These two sentences use exactly the same words the same number of times, but they say opposite things:

* The lecture was not good, but the notes were great.
* The lecture was good, but the notes were not great.

A Bag of Words model cannot tell them apart. The scikit learn guide gives a smaller example: a statement and a question with the same words get equal vectors, so "we lose the information that the last document is an interrogative form" [3].

### 3. Vocabulary size and sparsity

**Vocabulary size** is the number of different terms the vectorizer knows. It is the number of columns in the matrix. **Sparsity** is the share of cells that are zero.

**Why real matrices become large and mostly zero.** New words keep appearing as a collection grows. Jurafsky and Martin explain that the number of word types grows a little faster than the square root of the text length, and that a model keeps seeing unknown words [2]. At the same time, one document uses only a small part of the vocabulary. The scikit learn guide gives a concrete case: 10,000 short documents, such as emails, use about 100,000 different words in total, but each document uses only 100 to 1000 of them. So typically more than 99 percent of the cells are zero [3].

**How sparse matrices are stored.** Storing all the zeros would waste memory, so libraries keep only the non zero values and where they are. The common format is compressed sparse row (CSR) in SciPy. It uses three arrays: `data` holds the stored values, `indices` holds the column of each value, and `indptr` marks where each row starts [6]. SciPy lists efficient arithmetic, efficient row slicing and fast matrix vector products as strengths. Its weaknesses are slow column slicing and expensive changes to the structure [6]. CountVectorizer returns this kind of matrix.

**How settings change the vocabulary.** These descriptions come from the scikit learn documentation [4]:

| Setting | What it does to the vocabulary |
|:---|:---|
| `min_df` | Ignores terms that appear in fewer documents than the threshold. A whole number means a count, a decimal means a share of documents. It removes rare words and typing mistakes |
| `max_df` | Ignores terms that appear in more documents than the threshold. The documentation calls these corpus specific stop words |
| `max_features` | Keeps only the top terms ordered by how often they appear in the whole corpus |
| Lowercasing | Merges "Win" and "WIN" into one term. It is on by default |
| Stopwords | Removes the words on a list. The documentation warns that there are "several known issues" with the built in English list |
| Stemming | Merges forms such as "prizes" and "prize". It needs a stemmer from outside scikit learn |

The default tokenizer also shapes the vocabulary. Its pattern accepts only tokens of two or more letters or digits and treats punctuation as a separator [4]. So one letter words and signs like "!" never reach the vocabulary.

### 4. N grams

An n gram is a sequence of n words in a row [2]. A unigram is one word, a bigram is two words that follow each other, and a trigram is three. For the sentence "the lecture was not good", the bigrams are "the lecture", "lecture was", "was not" and "not good".

**How they keep part of the word order.** A bigram records which word came right after which. A trigram looks two words back. The scikit learn guide shows this with an example: the bigram "is this" appears only in the document that is a question, so the matrix can tell it apart from a statement with the same words [3]. The order is kept only inside the window of n words, and nothing is known about words further away.

**A bigram that changes the meaning.** The word "good" alone suggests a positive review. The bigram "not good" says the opposite. A unigram model sees "not" and "good" as two separate columns. A bigram model gets one column for "not good", so the negation stays attached to the word it changes. Part B, task 6 shows this on a review.

**The cost of larger n grams.** The number of possible n grams grows very fast. Jurafsky and Martin give an example with a vocabulary of 29,066 words: there are 844,000,000 possible bigrams, and about 7 times 10 to the power 17 possible 4 grams. They also say the resulting probability tables are "ridiculously sparse" [2]. In practice the costs are:

* A much larger vocabulary and a bigger, sparser matrix.
* More memory and slower training.
* Most n grams appear only once, so the model cannot learn much from them. This raises the risk of overfitting.
* More training data is needed to see each n gram often enough.

My own test in Part B shows the growth on a tiny corpus: 32 terms with unigrams, 67 with bigrams, 98 with trigrams.

### 5. TF IDF

TF IDF gives each word in a document a weight made of two parts. Term frequency (tf) says how often the word appears in the document. Inverse document frequency (idf) says how rare the word is in the whole collection. The textbook weight is their product [1]:

tf idf(t, d) = tf(t, d) times idf(t)

**Where the idea of idf came from.** Karen Spärck Jones introduced it in a 1972 paper in the Journal of Documentation. She argued that the specificity of a term can be measured statistically, by the number of documents the term appears in. A frequently used term is not specific, even if its meaning is exact, because it does not help to pick out the relevant documents [7]. Her first weight used a base 2 scale: if N is the number of documents and n is the number that contain the term, the weight was f(N) minus f(n) plus 1, where f(x) is the whole number m that fits 2 to the power (m minus 1) < x <= 2 to the power m. In her 200 document test collection, a term in 90 documents got weight 2 and a term in 3 documents got weight 7 [7]. She also found that dropping frequent terms hurt retrieval, so they should count for less, not be removed [7]. Robertson later noted that for years idf was seen as a heuristic with "no known or understood theoretical basis", and he showed how it can be justified with a probability argument [8].

**The textbook formula.** The IR book defines document frequency df as the number of documents that contain the term, and gives [1]:

idf(t) = log(N / df(t))

where N is the number of documents. The base of the logarithm does not matter for ranking [1].

**Why rarer words get higher weights.** If a word is in few documents, N divided by df is large, so its logarithm is large. If a word is in every document, N divided by df is 1, and the logarithm of 1 is 0. The book explains why this makes sense: a word like "auto" in a collection about the car industry says almost nothing, while a word that appears in only a few documents separates those documents from the rest [1]. The weight is highest when a term occurs many times in a small number of documents, and lowest when it occurs in nearly all documents [1].

**How scikit learn's `TfidfVectorizer` differs** [3][4]:

| | Textbook | scikit learn default |
|:---|:---|:---|
| tf | Count of the word in the document | The raw count (the same, unless `sublinear_tf` is turned on, which uses 1 plus log of the count) |
| idf | log(N / df) | log((1 + N) / (1 + df)) + 1 |
| Smoothing | None | Adds 1 to the top and bottom, as if one extra document held every word once. This avoids division by zero |
| The extra + 1 | None | Added outside the log, so a word found in every document keeps a weight above zero |
| Normalization | Not part of the formula | Each document vector is divided by its Euclidean (L2) length |

The scikit learn guide states that its formulas "differ slightly from the standard textbook notation", and it writes the textbook idf as log of n over (1 plus df) [3]. The IR book uses log of N over df [1]. So books also differ in small details, and a hand calculation will not match the library number.

**Why TF IDF values are weights, not probabilities.** A probability must describe the chance of something and the probabilities of all outcomes must add up to 1. TF IDF values do neither. An idf can be above 1 (scikit learn gives 2.10 for a word found in one of five documents in my test). After L2 normalization the squares of a document's values add up to 1, not the values themselves. A weight only says how much a word should count for a document. Robertson does show that idf can be read as the negative log of the chance that a random document contains the term [8], but that is a link to a probability, not a probability.

### 6. fit, transform and fit_transform, unknown words and data leakage

**`fit`** learns from the text it is given. For CountVectorizer that means building the vocabulary and deciding which word gets which column. For TfidfVectorizer it also learns the idf values. **`transform`** uses what was learned to turn any text into vectors, and it learns nothing new. **`fit_transform`** does both in one call on the same text, so it is only for the training data.

**Out of vocabulary words.** A word that was not in the fitted text has no column. The scikit learn guide says such words "will be completely ignored in future calls to the transform method" [3]. The word is lost without any warning, and nothing in the vector shows that it existed.

**Data leakage.** The scikit learn guide defines it like this: "Data leakage occurs when information that would not be available at prediction time is used when building the model." It leads to "overly optimistic performance estimates" and the general rule is "never call `fit` on the test data" [5]. In vectorization, leakage happens when the vectorizer is fitted on all the text, including the test set. The vocabulary then contains test words, and in TF IDF the idf values also use the test documents. Real new messages will contain words the vocabulary has never seen, but the test messages do not, so the test score looks better than the real performance. The scikit learn guide shows how big the bias can be: with random labels the true accuracy is about 0.5, but a leaky preprocessing step gave 0.76 [5]. The vectorizer is a smaller case of the same mistake.

**The correct order of steps.**

1. Split the data into a training set and a test set first.
2. Fit the vectorizer on the training text only, with `fit_transform`.
3. Train the model on those training vectors.
4. Use `transform` with the same fitted vectorizer on the test text. Never call `fit` again.
5. Predict and measure the score.

The same fitted vectorizer is then used on new messages when the model goes into use. A scikit learn `Pipeline` follows this order for you, also inside cross validation, and the guide recommends it for that reason [5].

### 7. Case study: comparing text representations on Swahili news

**Paper.** Kastanos, A. and Martin, T. (2021). *Graph Convolutional Network for Swahili News Classification.* arXiv:2103.09325 [9]. The text I read does not print a venue.

**Task.** Classify Swahili news articles into six classes: national, sports, entertainment, economy, international and health. The authors focus on the case where few labels and little computing power are available, which they describe as typical for low resource African languages.

**Data.** The Swahili News Classification dataset has 23,266 labelled articles. It is split 8:1:1 into 18,612 for training and 2,327 each for validation and testing. The classes are very uneven: the largest class has 8,193 training articles and the smallest has 687. So the main score is macro F1, which gives each class the same weight [9].

**Representations compared.** In every baseline the feature vectors are given to a logistic regression classifier [9].

1. TF IDF.
2. Counts, which is the raw word count per document, the same as Bag of Words.
3. fastText, which averages pre trained 300 dimension Swahili word vectors over the document.
4. PV DBOW and PV DM, two document embedding methods (doc2vec).
5. Text GCN, a graph neural network. Words and documents are nodes in one graph. Document to word links use TF IDF, and word to word links use how often words appear close to each other.
6. Text GCN t2v, the same graph with word2vec and doc2vec vectors as the starting features.

**Results** (test set, average over 5 runs) [9]:

| Representation | Accuracy (%) | Macro F1 (%) |
|:---|:---:|:---:|
| TF IDF | 83.07 | 68.72 |
| Counts | 83.32 | 73.60 |
| fastText average | 67.47 | 32.41 |
| PV DBOW | 81.64 | 72.93 |
| PV DM | 77.01 | 67.50 |
| Text GCN | 84.62 | 75.29 |
| Text GCN t2v | 85.40 | 75.67 |

Three things stand out. First, plain Counts and TF IDF are strong baselines: their accuracy is within about 2 points of the best model. Second, TF IDF did not beat plain Counts in macro F1 (68.72 against 73.60), which shows that a smarter weighting is not always better. Third, simply averaging pre trained word vectors was the worst by a wide margin (macro F1 of 32.41). Dense embeddings are not automatically better; how they are used matters. The best results came from the graph models, and the authors report that they beat the traditional methods most clearly when the share of labels is reduced, while TF IDF and Counts "see a significant degradation in performance" when the labels drop from 5 percent to 1 percent [9].

**How the study shows the main limitation of Bag of Words and TF IDF.** The paper does not say in those words that counts and TF IDF miss meaning, and I do not want to claim that it does. The limitation is visible in how the authors designed the study. Their reason for using a graph is that, although news text "superficially" looks like a sequence of words, "a document contains an implicit graph structure in the form of semantic and syntactic relationships" [9]. Counts and TF IDF treat each document alone, as a bag, and each word as an unrelated column. The graph adds the links between documents and words and between words that occur near each other, and that extra information is what lifts the score. In my reading, the gap between the best model and the TF IDF macro F1 (75.67 against 68.72) is the price of ignoring how words relate to each other. This matches what Jurafsky and Martin say about sparse count vectors: in them "dimensions for synonyms like car and automobile dimension are distinct and unrelated" [2].

**A supporting result for Somali.** The MasakhaNEWS study (Adelani et al., 2023) classifies news topics in 16 African languages, including Somali with 2,915 articles from BBC Somali [10]. Its classical models used scikit learn's CountVectorizer to represent text as token counts, with Naive Bayes, a multilayer perceptron and XGBoost [10]. On Somali, their weighted F1 scores were 74.8, 71.1 and 68.5. The pre trained multilingual models AfroXLMR large and AfriBERTa, which build meaning from context, reached 86.9 and 83.9. Averaged over all languages, the classical models scored 82.8 to 85.7, against 92.6 for AfroXLMR large [10]. The authors state that classical machine learning models "are worse in general than fine tuning multilingual LMs on average" [10].

**My comment.** The Swahili study uses only one dataset and one classifier for the baselines, and the exact numbers for fewer labels are only in a figure, so I did not use them. Even so, both studies point the same way: counts and TF IDF are cheap and surprisingly strong, but they stop improving where meaning and context are needed.

## Part B: Practical

All results below come from my script `vectorization_practice.py`. I ran it in Google Colab and saved the print out in `output.txt`.

### Task 2: the document term matrix

The vocabulary has 32 words and the matrix has the shape (5, 32). Each row is a document and each column is a word.

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

### Task 7: counts and TF IDF weights for D2

Counts, document frequency (df), idf and the TF IDF weight of every word in D2, from the TF IDF matrix:

| Word | Count | df | idf | TF IDF weight |
|:---|:---:|:---:|:---:|:---:|
| call | 1 | 1 | 2.10 | 0.39 |
| cash | 1 | 1 | 2.10 | 0.39 |
| entry | 1 | 1 | 2.10 | 0.39 |
| prizes | 1 | 1 | 2.10 | 0.39 |
| free | 1 | 2 | 1.69 | 0.31 |
| now | 1 | 2 | 1.69 | 0.31 |
| today | 1 | 2 | 1.69 | 0.31 |
| win | 1 | 2 | 1.69 | 0.31 |

### Task 8: TF IDF by hand for D2

D2 has 8 words by the vectorizer's tokenizer: free, entry, win, cash, prizes, today, call, now. The number of documents is N = 5. I used tf = count divided by the number of words in D2, and idf = log(N / df) with the natural log.

| | cash | today |
|:---|:---:|:---:|
| count in D2 | 1 | 1 |
| tf = count / 8 | 1 / 8 = 0.125 | 1 / 8 = 0.125 |
| df (documents that contain the word) | 1 | 2 |
| N / df | 5 / 1 = 5.0 | 5 / 2 = 2.5 |
| idf = log(N / df) | 1.6094 | 0.9163 |
| tf idf = tf times idf | 0.125 times 1.6094 = 0.2012 | 0.125 times 0.9163 = 0.1145 |
| scikit learn idf (smoothed) | 2.0986 | 1.6931 |
| scikit learn weight in D2 | 0.3891 | 0.3140 |

### Short answers

**Task 1: one hot encoding.** The vector for `free` is [0, 0, 1, 0, 0, 0] and for `cash` it is [1, 0, 0, 0, 0, 0]. The sentence `win free cash` gives 3 vectors of length 6, while `free cash` gave 2. The number of vectors depends on the sentence length because every word gets its own vector, so a model that needs one fixed input cannot use them as they are. Adding the vectors together gives one fixed length vector, which is a Bag of Words vector.

**Task 2: Bag of Words.** The value 2 is in D1 for the word `win`. D1 says "Win" and "WIN", and the vectorizer lowercases both and counts them as one word (D5 also has a 2, for `the`). The vectors for `win free phone` and `phone free win` are exactly the same (the script prints `Same vector: True`), so the order of the words is lost.

**Task 3: tokens and vocabulary size.** The five documents have 41 tokens in total but only 32 different words. Seven words appear in two documents: free, lecture, notes, now, the, today and win.

**Task 4: preprocessing inside the vectorizer.**
The letter `a` was dropped in all three vectorizers, because the default tokenizer keeps only tokens of two or more characters. The `!` was dropped too, because punctuation is only a separator. The default lowercases the text, so `Win` and `WIN` become one entry `win` (7 words in D1). With `lowercase=False` there are 8 entries and both `Win` and `WIN` stay, along with `FREE` and `Reply`. The stopword list removed `now` and `to` from D1. On D1 to D5 the word `not` was removed, because it is on the built in English list, so D5 becomes `lecture`, `good`, `notes`, `great`. The review now looks fully positive and the complaint is gone.

**Task 5: limiting the vocabulary.** The full vocabulary has 32 words. With `min_df=2` it falls to 7 words: free, lecture, notes, now, the, today and win. With `max_features=5` it falls to 5: free, lecture, now, the and win. The survivors are the words that appear often, which means common words such as `the` and a few spam words. Every word that appears in only one document, like `cash`, `claim` and `prizes`, is lost, and those rare words are the most specific ones. With `max_features=5`, `the` and `win` have a count of 3. Five words (free, lecture, notes, now, today) tie with a count of 2, and the vectorizer kept `free`, `lecture` and `now`.

**Task 6: n grams.** On D5 the vocabulary has 9 terms with `(1, 1)` and 18 with `(1, 2)`. The bigram `not good` keeps the reviewer's opinion, because it ties the negation to the word `good` (`was not` is the other bigram with `not`). On D1 to D5 the vocabulary has 32 terms with `(1, 1)`, 67 with `(1, 2)` and 98 with `(1, 3)`, which is 2.09 and 3.06 times the unigram size. It grows by about 30 to 35 terms for each step up in n, because almost every bigram and trigram in this tiny corpus appears only once. On a large corpus the growth is much bigger.

**Task 7: TF IDF.** In D2 every word appears once, so only idf makes a difference. `call`, `cash`, `entry` and `prizes` are in one document only, so they have the higher idf of 2.10 and the weight 0.39. `free`, `now`, `today` and `win` are in two documents, so their idf is 1.69 and their weight is 0.31. In D1, `win` has the highest weight, 0.57, because it appears twice. Its count of 2 times an idf of 1.69 is about 3.39, which is more than the 2.10 of a rare word that appears once, before the vector is normalized.

**Task 8: TF IDF by hand.** My results rank the two words the same way as scikit learn: `cash` is above `today` (0.2012 against 0.1145 by hand, 0.3891 against 0.3140 in scikit learn). The numbers are different because scikit learn uses a smoothed idf with 1 added (2.0986 and 1.6931) and normalizes the document vector, while my hand calculation does neither. The script prints `Same ranking: True`.

**Task 10: fit and transform.** After fitting on D1 to D4, N1 gives 1 for `claim`, `free` and `now`, and N2 gives 1 for `lunch`, `meeting`, `the` and `to`. The unknown words are `your` and `prize` in N1, and `is`, `moved` and `tomorrow` in N2. They were silently ignored. `prize` is unknown because the vocabulary holds the exact string `prizes`, and the vectorizer treats different strings as different words. Stemming or lemmatizing from Assignment 2 would fix it. The script shows that the Porter stemmer turns both `prizes` and `prize` into `prize`.

**Task 11: data leakage.** The clean vectorizer fitted on D1 to D4 has 26 words. The one fitted on D1 to D4 plus N1 and N2 has 31. The five words that leaked from the test texts are `is`, `moved`, `prize`, `tomorrow` and `your`. With the leaky vectorizer none of them would be unknown at test time, so the test score would look better than it really is.

**Task 12: no meaning.** The vocabulary is delicious, food, tasty, terrible, the, was, and each review is a vector over those six words. The cosine similarity of R1 with R2 is 0.75, and the cosine similarity of R1 with R3 is also 0.75, so `tasty` is not closer to `delicious` than to `terrible`. The vectors only share the words `food`, `the` and `was`. The three opinion words sit in three different columns, so the model sees no link between them. Counting words cannot show that two words mean similar things.

### Which vectorizer settings for a spam detector?

For a spam detector I would use TF IDF with its default L2 normalization, or plain counts, because both gave strong results in the case study. TF IDF also gave the repeated word `win` the top weight in D1 (0.57), and it gives the rare spam words such as `cash` and `prizes` the higher weights in D2 (0.39 against 0.31). I would keep lowercasing, since it merged `Win` and `WIN` into one entry (task 4), and I would add a separate count of capital words, as in Assignment 2. I would not use the built in `stop_words="english"` list. It removed `now` and `to` from D1, and `now` is a typical spam word, as in "call now", and it also removed `not` (task 4). Bigrams with `ngram_range=(1, 2)` are useful to keep short phrases, but the vocabulary doubled from 32 to 67 terms (task 6), so I would pair them with a low `min_df`. I would avoid `min_df=2` and `max_features=5` on a small data set, since they threw away `cash`, `claim` and `prizes`, which are the best spam clues (task 5). I would also check that punctuation such as `!` is not simply dropped, because the default tokenizer ignores it (task 4). Stemming would help with forms such as `prizes` and `prize` (task 10). Most importantly, I would fit the vectorizer only on the training messages and use `transform` on the test messages, because fitting on everything let five test words into the vocabulary (task 11).

## References

[1] Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. Online edition, chapter 6, sections "Term frequency and weighting", "Inverse document frequency" and "Tf idf weighting".

[2] Jurafsky, D. and Martin, J. H. (2025). *Speech and Language Processing*, 3rd edition draft (August 2025 draft). Chapter 2 (words, tokens and vocabulary), chapter 3 (N gram language models) and chapter 5 (Embeddings). https://web.stanford.edu/~jurafsky/slp3/

[3] scikit learn developers. User guide, "Feature extraction", section "Text feature extraction" (scikit learn 1.9.1). https://scikit-learn.org/stable/modules/feature_extraction

[4] scikit learn developers. API documentation for `CountVectorizer` and `TfidfTransformer` (`sklearn.feature_extraction.text`). https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

[5] scikit learn developers. "Common pitfalls and recommended practices", section on data leakage. https://scikit-learn.org/stable/common_pitfalls.html

[6] SciPy developers. Documentation of `scipy.sparse.csr_matrix`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

[7] Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11 to 21. https://www.doi.org/10.1108/EB026526

[8] Robertson, S. (2004). Understanding inverse document frequency: on theoretical arguments for IDF. *Journal of Documentation*, 60(5), 503 to 520.

[9] Kastanos, A. and Martin, T. (2021). Graph Convolutional Network for Swahili News Classification. arXiv preprint 2103.09325. https://arxiv.org/abs/2103.09325

[10] Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., et al. (2023). MasakhaNEWS: News Topic Classification for African Languages. arXiv preprint 2304.09972. https://arxiv.org/abs/2304.09972

All results in Part B come from `vectorization_practice.py` and `output.txt`.
