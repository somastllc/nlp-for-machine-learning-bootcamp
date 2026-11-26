# Assignment 3 – Vectorization

## Part A – Research

### 1. Text vectorization and one-hot encoding

Text vectorization means turning text into numbers so that a Machine Learning model can use it (Scikit-learn developers, n.d.). Models such as logistic regression learn one weight for each feature. This means every document must have the same number of features, in the same order. Then all documents fit in one table (rows = documents, columns = features), and each column means the same thing for every document.

One-hot encoding gives each word a vector with a single 1 and all other values 0. Real-life example: a spam filter for a phone shop with the vocabulary `[discount, lunch, offer, ticket]`.

| Word | discount | lunch | offer | ticket |
|---|---:|---:|---:|---:|
| `offer` | 0 | 0 | 1 | 0 |
| `ticket` | 0 | 0 | 0 | 1 |

One-hot encoding is not enough for a document because it gives one vector per word. The message "ticket offer" gives 2 vectors and "discount ticket offer" gives 3, so the size changes with the length of the message. The vectors also become very long when the vocabulary is big, and they show no similarity between words.

### 2. Bag of Words and the document-term matrix

Bag of Words (BoW) describes a document by counting how many times each vocabulary word appears in it (Jurafsky & Martin, 2024). It is called a "bag" because the words are counted but their order is forgotten, like items thrown in a bag.

In the document-term matrix, each **row** is a document, each **column** is a word from the vocabulary, and each **cell** is the number of times that word appears in that document.

| | lunch | meeting | offer | ticket | today |
|---|---:|---:|---:|---:|---:|
| ticket offer today | 0 | 0 | 1 | 1 | 1 |
| lunch meeting today | 1 | 1 | 0 | 0 | 1 |

Losing word order does little harm in spam detection, because words like `offer` and `ticket` are strong clues by themselves. It causes real problems when the order carries the meaning. "Dog bites man" and "man bites dog" have exactly the same bag, but they describe different events. In sentiment analysis, "the movie was good, not bad" and "the movie was bad, not good" also use the same words, but the opinion is the opposite.

### 3. Vocabulary size and sparsity

Vocabulary size is the number of different words the vectorizer keeps, which is the number of columns. Real vocabularies are very big because there are many rare words, names, spelling mistakes and word forms (walk, walks, walked). One document uses only a small part of them, so most cells are 0. This is called sparsity. In the small table above, 4 of the 10 cells are 0; in real datasets the share of zeros is often above 99%.

A sparse matrix stores only the non-zero values and their positions. For example, the CSR format keeps three arrays: `data` (the values), `indices` (the column of each value) and `indptr` (where each row starts) (SciPy developers, n.d.). Scikit-learn returns its vectorizer output as such sparse matrices (Scikit-learn developers, n.d.). Zeros are not stored, so memory is saved.

| Setting | What it does to the vocabulary |
|---|---|
| `min_df` | removes words that appear in too few documents (e.g. typos and rare names) |
| `max_df` | removes words that appear in too many documents, because a word found in almost every document does not help to tell documents apart |
| `max_features` | keeps only the N most frequent words |
| lowercasing | `Offer` and `offer` become one word |
| stopwords | removes common words like `the`; but a list can also remove useful words such as `not` |
| stemming | `walks`, `walked` become `walk`, so there are fewer words |

### 4. N-grams

An n-gram is a sequence of n neighboring words: a unigram is 1 word, a bigram is 2 words and a trigram is 3 words. They keep a small part of the word order, because the words inside the n-gram stay together (Jurafsky & Martin, 2024).

Example: in "the movie was not bad", the unigram `bad` looks negative, but the bigram `not bad` shows that the opinion is positive.

The cost: every new phrase becomes a new column, so the vocabulary grows fast. The matrix needs more memory and becomes more sparse, and rare phrases can cause overfitting.

| ngram_range | Number of features for "the movie was not bad" |
|---|---:|
| (1,1) | 5 |
| (1,2) | 9 |
| (1,3) | 12 |

### 5. TF-IDF

TF-IDF gives a high weight to a word that appears often in a document but in few documents overall. The idea of inverse document frequency (IDF) came from Karen Spärck Jones (1972), who explained that a rare word is more useful for finding documents than a common word.

Textbook formula (Manning et al., 2008, use the raw count as TF; dividing by the document length is a common variant, used here to keep the numbers small):

- TF = times the word appears in the document / number of words in the document
- IDF = log(N / df), where N = number of documents and df = number of documents that contain the word
- TF-IDF = TF × IDF

A rare word has a small df, so N / df is big and the IDF is high. A word that appears in all documents gets IDF = log(1) = 0.

Example with N = 3 documents: "ticket offer today", "lunch meeting today", "ticket price today". For the first document, each word has TF = 1/3 = 0.333.

| Word | df | IDF = log(3 / df) | TF-IDF |
|---|---:|---:|---:|
| `offer` | 1 | 1.099 | 0.366 |
| `ticket` | 2 | 0.405 | 0.135 |
| `today` | 3 | 0.000 | 0.000 |

**scikit-learn is different** (Scikit-learn developers, n.d.). It uses the raw count as TF, it uses a smoothed IDF = ln((1 + N) / (1 + df)) + 1, and then it normalizes each document vector with L2 normalization (the vector length becomes 1). The smoothing avoids division by zero, and the "+1" means a word found in every document is not set to 0. For the same example, scikit-learn gives `offer` 0.72, `ticket` 0.55 and `today` 0.43, so `today` is not zero. The ranking of the words is the same as the textbook ranking.

TF-IDF values are weights, not probabilities. They do not measure the chance of anything, and they do not add up to 1 (after L2 normalization, the squares of the values add up to 1, not the values themselves). A value of 0.72 only means that the word is more important in that document compared with other words.

### 6. fit, transform and fit_transform

- `fit` learns the vocabulary (and, for TF-IDF, the document frequencies) from the training text.
- `transform` converts text into vectors using what was already learned. It never adds new columns.
- `fit_transform` does both steps on the same text. It is used on training data.

Out-of-vocabulary (OOV) words are words that were not in the training text. They have no column, so they are ignored. Example: fit on "ticket offer" and "lunch meeting" (vocabulary: lunch, meeting, offer, ticket). The text "offer tomorrow" becomes `[0, 0, 1, 0]`, because `tomorrow` is unknown.

Data leakage means that information from the test data is used during training. If the vectorizer is fitted on train and test text together, the test words enter the vocabulary, and for TF-IDF the IDF values are also calculated with the test documents. The test data is then no longer completely new for the pipeline, so the score looks better than it would on real new data.

Correct order: split the data → fit the vectorizer on the training text only → transform both the training and the test text with the same vectorizer.

### 7. Case study

**Study:** Madodonga, Marivate and Adendorff (2023), *Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati*.

**Task:** classify news into topics (for example politics, education, crime) in isiZulu and Siswati, two African languages.

**Data:** 752 isiZulu news items and 80 Siswati headlines. After removing categories with very few examples, 563 isiZulu items and 68 Siswati headlines remained. Because the data was small, the authors also used data augmentation and SMOTE.

**Representations compared:** Bag of Words, TF-IDF and Word2vec, used with Logistic Regression, Naive Bayes, XGBoost and LSTM.

**Results** (F1-score, isiZulu articles, original data, Logistic Regression):

| Representation | F1 (%) |
|---|---:|
| Bag of Words | 36.06 |
| TF-IDF | 24.19 |
| Word2vec | 65.13 |

The best result on the original data came from Word2vec with LSTM (82.78%), and the best overall result was Word2vec with XGBoost after augmentation (95.21%). After augmentation, Bag of Words (83.09%) and TF-IDF (79.20%) with Logistic Regression improved a lot, but Word2vec with Logistic Regression was still higher (91.30%).

**Main limitation:** Bag of Words and TF-IDF treat every word as a separate column, so they do not know that two different words can have a similar meaning. Word2vec places words with similar meaning close together, and the authors even used this to replace words with similar ones when creating more data. This is a plausible reason why Word2vec did better in this study. However, the authors do not prove this: they say that the poor performance of Bag of Words and TF-IDF still needs more investigation.

## Part B – Practical

All numbers below come from my own script `vectorization_practice.py`; the full printed output is in `output.txt`.

### Task 2 – Document-term matrix (D1–D5, shape 5 × 32)

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

### Task 7 – Count vs TF-IDF weight for every word in D2

D2: "Free entry: win cash prizes today. Call now."

| Word | Count | TF-IDF weight |
|---|---:|---:|
| `call` | 1 | 0.389 |
| `cash` | 1 | 0.389 |
| `entry` | 1 | 0.389 |
| `prizes` | 1 | 0.389 |
| `free` | 1 | 0.314 |
| `now` | 1 | 0.314 |
| `today` | 1 | 0.314 |
| `win` | 1 | 0.314 |

### Task 8 – TF-IDF by hand for D2

The vectorizer finds 8 words in D2, so TF = 1 / 8 = 0.125 for both words. N = 5 documents, and df comes from the count matrix.

| Word | df | TF | IDF = ln(5 / df) | TF-IDF |
|---|---:|---:|---:|---:|
| `cash` | 1 | 0.125 | ln(5 / 1) = 1.609 | 0.125 × 1.609 = **0.201** |
| `today` | 2 | 0.125 | ln(5 / 2) = 0.916 | 0.125 × 0.916 = **0.115** |

scikit-learn gave `cash` = 0.389 and `today` = 0.314.

### Short answers

**Task 1 – One-hot encoding.** `free` gives `[0, 0, 1, 0, 0, 0]` and `cash` gives `[1, 0, 0, 0, 0, 0]`. `win free cash` gives 3 vectors, while `free cash` gives 2. One-hot encoding makes one vector per word, so the number of vectors grows with the length of the sentence. A model needs one vector of fixed size for each document.

**Task 2 – Bag of Words.** The value 2 in D1 is in the `win` column, because D1 contains "Win" and "WIN", and the vectorizer lowercases both to `win`. The vectors of `win free phone` and `phone free win` are the same (`Same vector? True`). Bag of Words counts words and forgets their order.

**Task 4 – Preprocessing.** In D1, the default vectorizer dropped `a` (one-letter words are ignored) and `!` (punctuation is removed), and it turned "Win" and "WIN" into one word `win`. With `lowercase=False`, `Win` and `WIN` become two separate words. The stopword list removed `now` and `to`. On D1–D5, `not` was removed (`Was 'not' removed? True`). This changes D5, "The lecture was not good, but the notes were great", into a text that contains `good` without `not`, so the negative opinion looks positive.

**Task 5 – Limiting the vocabulary.** The full vocabulary has 32 words. `min_df=2` keeps 7 words (`free, lecture, notes, now, the, today, win`) and `max_features=5` keeps 5 (`free, lecture, now, the, win`). The words that survive are the frequent ones, including common words like `the` and `now`, while rare but informative words such as `cash`, `phone` and `prizes` are lost. With `max_features`, several words had the same count (2), so which of them were kept is not guaranteed by scikit-learn.

**Task 6 – N-grams.** On D5, the bigram `not good` keeps the opinion of the reviewer: it stays together, while the unigram `good` alone looks positive. On D1–D5 the vocabulary size grows from 32 (1,1) to 67 (1,2) to 98 (1,3). The vocabulary roughly doubles when bigrams are added, and trigrams add about 31 more, so the growth is fast even for 5 short documents.

**Task 7 – TF-IDF.** In D2 every word appears once, so TF is the same for all words. `call`, `cash`, `entry` and `prizes` appear only in D2 (df = 1) and get 0.389, while `free`, `now`, `today` and `win` appear in two documents (df = 2) and get the lower weight 0.314. In D1, `win` gets the highest weight (0.574), but not because it is rare: it appears in two documents, like `free` and `now` (0.287). It is highest because it appears **twice** in D1, and 2 × 0.287 = 0.574.

**Task 8 – TF-IDF by hand.** My hand result is `cash` = 0.201 and `today` = 0.115, so `cash` is higher, the same ranking as scikit-learn (0.389 and 0.314). The numbers are different because scikit-learn uses the raw count, a smoothed IDF and L2 normalization.

**Task 10 – Fit and transform.** The vectorizer was fitted on D1–D4 (26 words). The out-of-vocabulary words are `your` and `prize` in N1, and `is`, `moved` and `tomorrow` in N2; they have no column and are ignored. `prize` is unknown because D2 contains `prizes`, which is a different word form. Stemming or lemmatization from Assignment 2 would turn `prizes` into `prize`.

**Task 11 – Data leakage.** Fitting on D1–D4 + N1 + N2 added 5 words to the vocabulary that only exist in the test texts: `is`, `moved`, `prize`, `tomorrow` and `your`. This is leakage, because the test set changed the vocabulary before evaluation. The correct way is to fit only on the training texts.

**Task 12 – No meaning.** The cosine similarity of R1 (`tasty`) with R2 (`delicious`) is 0.75, and with R3 (`terrible`) it is also 0.75, so R1 is **not** more similar to R2. Each review shares 3 of its 4 words with the others, and `tasty`, `delicious` and `terrible` are three separate columns. Bag of Words only matches identical words, so it cannot see that `tasty` and `delicious` have almost the same meaning.

### Which settings for a spam detector?

For a spam detector I would use TF-IDF (or counts) with unigrams and bigrams, lowercasing, a modest `min_df`, and no default stopword list. My outputs show why. The scikit-learn stopword list removed `now` and `to` from D1 and `call` and `now` from the full corpus, words that appear in spam messages like "Call now", and it also removed `not` from D5. Bigrams such as `free prize` or `not good` keep information that single words lose, but they make the vocabulary grow fast (32 → 67 → 98 on only 5 documents), so I would limit them with `min_df` on a large dataset. TF-IDF gave higher weights to words that appear in fewer documents (`cash`, `entry`, `prizes`, 0.389) than to common ones (`free`, `now`, 0.314), which is useful. I would avoid a very small `max_features` or too strict `min_df`: with `max_features=5` the word `the` was kept while `cash` and `phone` were lost. I would also avoid using the default stopword list without checking it, and I would avoid fitting the vectorizer on test messages (Task 11), because the evaluation would then look better than it really is.

## References

Jurafsky, D., & Martin, J. H. (2024). *Speech and language processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

Madodonga, A., Marivate, V., & Adendorff, M. (2023). Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati. *Journal of the Digital Humanities Association of Southern Africa, 4*(01). https://doi.org/10.55492/dhasa.v4i01.4449

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

SciPy developers. (n.d.). scipy.sparse.csr_matrix. https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

Scikit-learn developers. (n.d.). Text feature extraction. https://scikit-learn.org/stable/modules/feature_extraction.html

Scikit-learn developers. (n.d.). TfidfVectorizer. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21.
