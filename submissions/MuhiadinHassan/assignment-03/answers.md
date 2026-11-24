# Assignment 3 – Vectorization

## Part A – Research

### 1. Text vectorization and one-hot encoding

Text vectorization maps text into numerical feature vectors so that a learning algorithm can calculate distances, weights, or decision boundaries from it. Many conventional machine-learning models expect a fixed-width input: each position must refer to the same feature for every example. If one email has 12 features and another has 15, a model cannot compare their coordinates consistently. A text vocabulary supplies a shared set and order of features.

With one-hot encoding, each vocabulary word receives its own position. The word is represented by a vector of zeroes with a single 1 at its position. In a spam filter with vocabulary `[cash, class, free, lunch, notes, win]`, `free` is `[0,0,1,0,0,0]`. A sentence is represented by one vector per token, so its length varies; one-hot encoding alone does not make a single fixed-width sentence vector. It also treats every word as unrelated to every other word and does not encode word order or frequency. A document-term matrix or a sequence model can turn a document into a fixed-size input.

### 2. Bag of Words and document-term matrices

Bag of Words (BoW) represents a document by the counts of vocabulary terms it contains. In a document-term matrix, rows are documents, columns are vocabulary terms, and each cell is the count for that word in that document. It is called a “bag” because the representation ignores the order in which words appeared. In the five-document practice corpus, D1 has count 2 for `win`, because both “Win” and “WIN” are lowercased to the same feature.

Discarding order can work reasonably for topic categorization or spam detection, where term presence and frequency often help. It can cause serious errors in sentiment or intent: “the food was good” and “the food was not good” share most words, but have opposite meanings. N-grams restore some local order, while more advanced representations can model context.

### 3. Vocabulary size and sparsity

The vocabulary size is the number of distinct features learned from the corpus. The resulting matrix can be wide because every distinct word (or n-gram) gets a column, yet a document uses only a small portion of all possible terms. Most cells are therefore zero. Sparse formats such as compressed sparse row (CSR) store the nonzero values and their column positions instead of every zero, reducing memory and often speeding up operations.

Vectorizer choices change the vocabulary. `min_df` drops terms appearing in fewer than a chosen number of documents; `max_df` drops terms occurring in too many documents; `max_features` keeps only a limited set of frequent terms. Lowercasing merges case variants, stopword filtering removes selected common terms, and stemming can merge related word forms. These operations can reduce size, but may remove useful distinctions or change meaning. In the practice corpus, the full unigram vocabulary contains 32 terms; `min_df=2` leaves 7, while `max_features=5` leaves 5.

### 4. N-grams

An n-gram is a contiguous sequence of *n* tokens. Unigrams are single words, bigrams are two adjacent words, and trigrams are three. For example, the bigram “not good” preserves a negation that independent unigram counts can obscure. N-grams still do not represent full sentence structure or word meaning, but they retain local order.

Larger n-gram ranges can expand the feature space quickly, make the matrix sparser, and increase memory and overfitting risks. In this exercise the vocabulary grows from 32 unigrams to 67 unigram-plus-bigram features, then to 98 features with trigrams as well. N-grams also become less reusable when a phrase appears in only one document.

### 5. TF-IDF and inverse document frequency

TF-IDF combines a term’s frequency in a document (TF) with its inverse document frequency (IDF), which discounts terms spread across many documents. The textbook form used here is:

\[
\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\times\log\left(\frac{N}{\mathrm{df}(t)}\right)
\]

where `N` is the number of documents and `df(t)` is the number of documents containing term *t*. Karen Spärck Jones formalized IDF as a way of representing a term’s specificity within a collection: terms appearing in fewer documents can help distinguish one document from another. For the tiny example, `cash` appears in one of five documents, while `today` appears in two, so the textbook IDF for `cash` is higher.

Scikit-learn’s default `TfidfVectorizer` uses smoothed IDF, `log((1+N)/(1+df))+1`, and by default L2-normalizes each document vector. Its output is therefore not the same scale as the textbook calculation. The values are feature weights, not probabilities: they need not sum to 1, and they do not describe mutually exclusive outcomes. They quantify relative term importance under a specified formula and corpus.

### 6. `fit`, `transform`, and leakage

`fit` learns the vocabulary and any statistics needed from text. `transform` applies that already learned mapping to new text; words not in the fitted vocabulary are out-of-vocabulary (OOV) and do not add new columns. `fit_transform` learns and transforms the same data in one operation. In an evaluation split, fit preprocessing/vectorization on training data only, then transform validation or test text using that fitted object.

Data leakage occurs if test text influences the vocabulary or other preprocessing statistics before evaluation. This gives the model information it would not have at deployment and can make evaluation misleadingly optimistic. Even an unsupervised vocabulary fitted on test examples leaks information about which terms exist in the evaluation set. Use a pipeline and fit it only inside each training fold; then transform the held-out fold.

### 7. Case study: Swahili news classification

Kastanos and Martin studied categorizing Swahili news articles into six classes. Their paper reports 23,266 labeled articles and compares lexical count and TF-IDF representations, averaged pretrained fastText word embeddings, paragraph-vector methods, and graph-based text representations. In a low-label experiment using 20% of training labels, TF-IDF achieved 83.07% accuracy and 68.72 macro-F1; count features achieved 83.32% accuracy and 73.60 macro-F1. Their TextGCN variants achieved higher macro-F1 (75.29 and 75.67, respectively) in that reported setting.

This is a text-classification study and demonstrates that simple lexical vectors can be useful, but their performance depends on available labels and representation choice. BoW and TF-IDF create separate dimensions for surface words, so synonyms such as two different words for a similar concept do not automatically share a feature. The graph-based methods add relationships among words and documents, which can supply information that a plain frequency vector lacks. The result does not imply that one representation always wins; it shows that the task and amount of labeled data matter.

## Part B – Practical

The tables and reported values below were generated by `vectorization_practice.py`; the full console output is in `output.txt`. The matrix columns follow the vectorizer’s alphabetical feature order.

### Task 2: Document-term matrix

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

The shape is 5 documents × 32 features. In D1, `win` has value 2 because the default analyzer lowercases both “Win” and “WIN” and counts both occurrences. The two transformed sentences “win free phone” and “phone free win” have the same vector because the count representation ignores word order.

### Task 7: D2 counts and TF-IDF

| Word | Count in D2 | TF-IDF weight in D2 |
|---|---:|---:|
| call | 1 | 0.389141 |
| cash | 1 | 0.389141 |
| entry | 1 | 0.389141 |
| free | 1 | 0.313957 |
| now | 1 | 0.313957 |
| prizes | 1 | 0.389141 |
| today | 1 | 0.313957 |
| win | 1 | 0.313957 |

Although each D2 term occurs once, terms that occur in fewer corpus documents receive a larger inverse-document weight. `call`, `cash`, `entry`, and `prizes` occur in only D2; `free`, `now`, `today`, and `win` appear in multiple documents. Scikit-learn then normalizes the entire row, so the displayed weight is also affected by the other terms in D2. In D1, `win` occurs twice and has weight 0.573889, the largest in that row.

### Task 8: Hand calculation for D2

There are `N=5` documents. `cash` occurs once in D2 and has `df=1`; `today` occurs once in D2 and has `df=2` (D2 and D3).

| Term | TF | DF | IDF `ln(N/df)` | Textbook TF-IDF | scikit-learn L2 weight |
|---|---:|---:|---:|---:|---:|
| cash | 1 | 1 | `ln(5/1) = 1.609438` | 1.609438 | 0.389141 |
| today | 1 | 2 | `ln(5/2) = 0.916291` | 0.916291 | 0.313957 |

Both methods rank `cash` above `today`. The actual values differ because the hand calculation uses textbook IDF without smoothing and without row normalization, while the vectorizer smooths IDF and normalizes each document vector.

### Short answers to the practical tasks

#### Task 1

The custom vocabulary assigns one six-value vector to each word: `free` is `[0,0,1,0,0,0]`, while `cash` is `[1,0,0,0,0,0]`. “win free cash” produces three vectors because one-hot encoding represents each token separately; therefore, the number of vectors follows the sentence’s token count. By itself this is not a fixed-width sentence representation.

#### Task 2

`CountVectorizer` learned 32 features and produced a 5×32 matrix. D1’s `win` cell is 2 due to two case variants becoming lowercase `win`. The two queries produce identical vectors because bag-of-words records counts, not positions; this is useful for order-insensitive tasks but loses word sequence.

#### Task 4

The default lowercases text, so `Win` and `WIN` merge into `win`; with `lowercase=False`, the features `WIN` and `Win` remain distinct. `a` is not included because the default token pattern requires at least two word characters, and `!` is punctuation, not a token. The English stopword list contains `a`, `to`, and `now`, but only `to` and `now` were actual D1 vocabulary features removed (the one-letter `a` was already filtered out). Across the full corpus, the stop list also removes `not`, so D5 becomes `lecture good notes great` and loses the negation that changes “not good” into a negative opinion.

#### Task 5

The full vocabulary has 32 words, `min_df=2` keeps 7 words found in at least two documents, and `max_features=5` keeps only five high-frequency terms. The surviving terms favor repeated or frequent vocabulary over rare terms; thus, a rare but useful spam indicator could be discarded by an aggressive limit.

#### Task 6

Unigrams represent `not` and `good` independently, while the bigram `not good` preserves their local relationship. On all five documents, the feature count grows from 32 unigrams to 67 unigram/bigram features and 98 features through trigrams. Larger ranges retain more word order but expand the feature space and sparsity.

#### Task 7

The TF-IDF matrix reweights terms based on their document frequency and then normalizes each row. In D2, every term occurs once, but terms unique to that document have higher weights than repeated corpus terms. In D1, `win` occurs twice and gets weight 0.573889, the highest in that document.

#### Task 8

The hand calculation gives `cash` an IDF of 1.609438 and `today` 0.916291, so textbook TF-IDF ranks cash higher. Scikit-learn’s corresponding normalized values are 0.389141 and 0.313957, preserving that ranking while using smoothed IDF and L2 normalization.

#### Task 10

When the vectorizer is fit on D1–D4, N1’s OOV terms are `prize` and `your`; N2’s are `is`, `moved`, and `tomorrow`. `prize` is unknown because the learned vocabulary contains the exact token `prizes`, not its singular form. Stemming or lemmatization before fitting and transforming can normalize related word forms, although it cannot solve every unseen-word case.

#### Task 11

Fitting only on D1–D4 creates the training vocabulary. If N1 and N2 are included during fitting, `is`, `moved`, `prize`, `tomorrow`, and `your` enter the vocabulary even though they came only from test data. This is leakage; fit on training data alone, then call `transform` on held-out data.

#### Task 12

The cosine similarity of R1 with R2 is 0.75, and R1 with R3 is also 0.75. All three reviews share `the food was`, and each has a different single opinion word, which receives an independent feature; ordinary BoW therefore treats the positive `tasty` and `delicious` and negative `terrible` words symmetrically. This illustrates that counts capture lexical overlap, not semantic similarity or polarity.

### Spam detector design choice

For a first spam detector I would compare word counts with TF-IDF using a held-out validation set, and also test unigrams plus bigrams so that phrases such as “claim prize” or negation are not reduced entirely to isolated words. I would avoid automatically removing all English stopwords: in this output, that setting removes `not` and changes D5’s “not good” into the remaining terms “good” and “great,” potentially losing sentiment and intent cues. I would use vocabulary limits only after measuring their effect, since `max_features=5` discards many potentially useful rare spam terms; `min_df` can control noisy one-off features, but should be tuned. Any vectorizer must be fit on training data only, then used to transform validation and test messages.

## References

1. Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. [https://doi.org/10.1108/eb026526](https://doi.org/10.1108/eb026526)
2. scikit-learn developers. *Feature extraction: text feature extraction and TF-IDF*. [https://scikit-learn.org/stable/modules/feature_extraction.html](https://scikit-learn.org/stable/modules/feature_extraction.html)
3. scikit-learn developers. *CountVectorizer API reference*. [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
4. scikit-learn developers. *TfidfVectorizer API reference*. [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
5. scikit-learn developers. *Common pitfalls and recommended practices: data leakage*. [https://scikit-learn.org/stable/common_pitfalls.html](https://scikit-learn.org/stable/common_pitfalls.html)
6. Kastanos, A., & Martin, T. (2021). *Graph Convolutional Network for Swahili News Classification*. arXiv:2103.09325. [https://arxiv.org/abs/2103.09325](https://arxiv.org/abs/2103.09325)
