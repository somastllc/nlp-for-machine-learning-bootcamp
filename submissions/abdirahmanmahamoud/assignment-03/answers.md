# Assignment 3 – Vectorization

## Part A – Research

### 1. Text vectorization and one-hot encoding

Text vectorization converts symbolic text into numerical feature vectors that machine-learning algorithms can process. A conventional model expects a fixed feature space: every position must have the same meaning in every example, and each vector must have the same length. For example, if column 4 represents “lunch”, it must represent “lunch” for every document; otherwise comparisons and learned weights would be inconsistent.

One-hot encoding assigns each vocabulary item a vector of zeros with a single 1 at its own index. With vocabulary `[apple, bus, tea]`, “bus” is `[0, 1, 0]`. This encodes one word only. A whole document such as “tea is cold” contains several tokens and cannot be represented by just one token's one-hot vector; a document representation must combine the vectors, for example by summing them into word counts.

### 2. Bag of Words and the document-term matrix

Bag of Words (BoW) represents each document by the occurrence or count of vocabulary terms. In a document-term matrix, rows represent documents, columns represent vocabulary terms, and each cell gives the count of that term in that document (or 0/1 in a binary version).

It is called a “bag” because word order and grammar are discarded: the model keeps the items and their counts, not their sequence. This can work for broad topic classification, such as sorting news into sports or business, where characteristic words are useful. It can fail for sentiment or intent: “the food is good” and “the food is not good” have nearly the same unigram counts even though the meanings differ.

### 3. Vocabulary size and sparsity

The vocabulary is the set of distinct features learned from the corpus; vocabulary size is the number of retained features. Real corpora contain many documents, rare names, spelling variants, inflections, and domain-specific terms, but each document uses only a small fraction of all corpus terms. Therefore, most cells in a document-term matrix are zero: the matrix is sparse.

Sparse formats such as compressed sparse row (CSR) store non-zero values and their column positions, plus row pointers, instead of storing every zero. This saves memory and can speed up computation when most cells are zero. `min_df` removes terms occurring in too few documents; `max_df` removes terms occurring in too many; `max_features` caps the vocabulary using corpus term frequency. Lowercasing merges case variants, stopword removal deletes selected common terms, and stemming combines related word forms. These choices reduce feature count but can also remove useful distinctions.

### 4. N-grams

An n-gram is a contiguous sequence of n tokens. Unigrams are single words, bigrams are two adjacent words, and trigrams are three adjacent words. For “not very useful”, bigrams include “not very” and “very useful”; the trigram is “not very useful”. N-grams retain some local order that ordinary unigram BoW loses.

A bigram such as “not good” preserves a negation cue that the unigram “good” alone cannot express. Larger n-grams create many more possible features, increase memory and processing costs, and often occur rarely. They can overfit small datasets and may not appear in future messages, so their value should be tested on held-out data.

### 5. TF-IDF and inverse document frequency

TF-IDF multiplies term frequency (TF), a measure of how often a term occurs in one document, by inverse document frequency (IDF), a measure that downweights terms appearing in many documents. A common textbook definition is:

\[
\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\times\mathrm{idf}(t), \qquad
\mathrm{idf}(t)=\log\left(\frac{N}{\mathrm{df}(t)}\right)
\]

Here, `N` is the number of documents and `df(t)` is the number containing term `t`. A term found in nearly every document is not useful for distinguishing documents, so its IDF is low. A term found in relatively few documents receives a higher IDF.

The historical basis of IDF is Karen Spärck Jones's 1972 paper, “A Statistical Interpretation of Term Specificity and Its Application in Retrieval.” She argued that term specificity can be estimated from how widely terms occur in a collection: rare terms can be more discriminative than very common ones.

Scikit-learn's default `TfidfVectorizer` smooths IDF:

\[
\mathrm{idf}\_{sklearn}(t)=\log\left(\frac{1+N}{1+\mathrm{df}(t)}\right)+1
\]

It multiplies term frequency by this IDF and applies L2 normalization to each document vector by default. Raw term counts are used unless `sublinear_tf=True`, which applies logarithmic TF scaling. Smoothing prevents extreme values and zero divisions. TF-IDF values are weights, not probabilities: they are not probabilities of a word or event and are not required to sum to one as a probability distribution.

### 6. `fit`, `transform`, and data leakage

`fit` learns the vocabulary and, for TF-IDF, document-frequency statistics from supplied training text. `transform` applies those learned rules to new text without changing the feature space. `fit_transform` performs both operations on the same data, generally more efficiently. Out-of-vocabulary (OOV) terms are ignored during transformation because no column was learned for them.

Data leakage occurs when information from the test set influences preprocessing or model fitting. Fitting a vectorizer on both training and test text lets test-only terms and their document frequencies affect the representation before evaluation. This weakens the independence of the test set and can make performance estimates overly optimistic. The correct order is: split first; fit preprocessing/vectorization only on training data; transform training and test data with that fitted vectorizer; train on training features; evaluate on untouched test features. In cross-validation, fit the vectorizer inside each training fold, preferably through a scikit-learn `Pipeline`.

### 7. Somali-language case study

Ali (2026), in _Scientific Reports_, studied Somali news topic classification using 7,240 manually curated articles collected from BBC Somali and VOA Somali. The articles were grouped into eight categories: health, sports, conflict, religion, business, entertainment, politics, and technology. The system used word-level TF-IDF unigram features with a linear Support Vector Machine. The paper reports 94.20% accuracy on a stratified 80/20 holdout split, with repeated holdout and five-fold cross-validation means around 93%.

This case demonstrates that a relatively simple representation can be an effective and resource-efficient baseline for Somali, a language with fewer established NLP resources than English. However, unigram TF-IDF represents lexical frequency and document distribution, not meaning. It does not inherently recognize synonyms, contextual meaning, or word order; semantically similar expressions with different words may have little vector overlap. Strong results on a particular news dataset show that word patterns can predict its categories, not that TF-IDF understands Somali.

## Part B – Practical

The following tables are generated from the same scikit-learn corpus operations used in `vectorization_practice.py`. The complete step-by-step console output is in `output.txt`.

### Task 2 – Document-term matrix

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me  | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to  | today | was | we  | were | win |
| -------- | --- | ------ | --- | ---- | ---- | ----- | ----- | ----- | --- | ---- | ---- | ----- | ------- | ----- | --- | ------- | --- | ----- | --- | ----- | ------ | ------ | ----- | ---- | ----- | --- | --- | ----- | --- | --- | ---- | --- |
| D1       | 0   | 0      | 0   | 0    | 0    | 1     | 0     | 0     | 0   | 1    | 0    | 0     | 0       | 0     | 0   | 0       | 0   | 0     | 1   | 1     | 0      | 0      | 1     | 0    | 0     | 0   | 1   | 0     | 0   | 0   | 0    | 2   |
| D2       | 0   | 0      | 0   | 1    | 1    | 0     | 0     | 1     | 0   | 1    | 0    | 0     | 0       | 0     | 0   | 0       | 0   | 0     | 1   | 0     | 0      | 1      | 0     | 0    | 0     | 0   | 0   | 1     | 0   | 0   | 0    | 1   |
| D3       | 1   | 0      | 0   | 0    | 0    | 0     | 0     | 0     | 1   | 0    | 0    | 0     | 0       | 1     | 0   | 1       | 0   | 0     | 0   | 0     | 0      | 0      | 0     | 0    | 1     | 0   | 0   | 1     | 0   | 1   | 0    | 0   |
| D4       | 0   | 1      | 0   | 0    | 0    | 0     | 1     | 0     | 0   | 0    | 0    | 0     | 1       | 0     | 1   | 0       | 0   | 1     | 0   | 0     | 1      | 0      | 0     | 1    | 0     | 1   | 0   | 0     | 0   | 0   | 0    | 0   |
| D5       | 0   | 0      | 1   | 0    | 0    | 0     | 0     | 0     | 0   | 0    | 1    | 1     | 1       | 0     | 0   | 0       | 1   | 1     | 0   | 0     | 0      | 0      | 0     | 0    | 0     | 2   | 0   | 0     | 1   | 0   | 1    | 0   |

### Task 7 – D2 count and TF-IDF comparison

| Word   | D2 count | D2 TF-IDF |
| ------ | -------: | --------: |
| call   |        1 |    0.3891 |
| cash   |        1 |    0.3891 |
| entry  |        1 |    0.3891 |
| free   |        1 |    0.3140 |
| now    |        1 |    0.3140 |
| prizes |        1 |    0.3891 |
| today  |        1 |    0.3140 |
| win    |        1 |    0.3140 |

### Task 8 – Hand calculations

For this calculation, TF is the raw count divided by the total number of vectorizer tokens in D2, and textbook IDF is `ln(N/df)` with `N = 5`.

| Term  | Count | D2 tokens |       TF |  df |   N | ln(N/df) |   TF-IDF |
| ----- | ----: | --------: | -------: | --: | --: | -------: | -------: |
| cash  |     1 |         8 | 0.125000 |   1 |   5 | 1.609438 | 0.201180 |
| today |     1 |         8 | 0.125000 |   2 |   5 | 0.916291 | 0.114536 |

The textbook calculation is not expected to match scikit-learn's numerical values exactly: scikit-learn smooths IDF and applies L2 normalization. Compare the relative term weights in Task 7 with the raw textbook scores above.

### Short answers to practical tasks

**1. One-hot encoding.** Each vocabulary word is a six-element vector with one `1` at its own index and zeros elsewhere. “win free cash” produces three word vectors because it contains three tokens; one-hot encoding represents individual vocabulary items, not a complete sentence in one vector.

**2. Bag of Words.** The DTM rows are documents and the columns are vocabulary terms; each cell is the word count. D1 has a count of 2 for `win` because it appears twice, and the two word-order permutations produce the same count vector because BoW ignores order.

**4. Preprocessing.** The default vectorizer lowercases text and ignores punctuation; its default token pattern also excludes one-character tokens such as `a`. With `lowercase=False`, `Win` and `WIN` remain separate case-sensitive features. English stopword removal removes common words and also removes `not`, which can turn D5's “not good” into the misleading remaining feature “good”.

**5. Vocabulary limits.** `min_df=2` retains terms appearing in at least two documents and removes one-document terms. `max_features=5` keeps only five of the most frequent features, which may discard rare but predictive words; the exact vocabularies and sizes are printed in `output.txt`.

**6. N-grams.** Unigrams represent individual terms; bigrams add adjacent two-word phrases. In D5, `not good` preserves the reviewer's negative opinion; adding bigrams and trigrams increases the number of features and therefore memory use and sparsity.

**7. TF-IDF.** TF-IDF reduces the influence of terms spread across many documents and emphasizes terms that distinguish a document. In D2 all terms occur once, but their weights differ because their document frequencies differ; in D1, `win` has a high weight because it occurs twice and is not common across the corpus.

**8. Hand TF-IDF.** The calculation uses the textbook formula `TF × ln(N/df)`. It will not numerically equal the scikit-learn output because the library uses smoothed IDF and L2 normalization; the exact values are shown in the hand-calculation table and console output.

**10. Fit and transform.** The vectorizer learns its vocabulary from D1–D4 and maps N1/N2 into those existing columns. Test words absent from that vocabulary are OOV; `prize` differs from `prizes`, and stemming or lemmatization can normalize related forms if the same preprocessing is applied consistently to both training and test text.

**11. Data leakage.** Fitting on D1–D4 plus N1 and N2 allows test-only words into the learned vocabulary. Those leaked terms are the difference between the combined and training-only vocabularies printed in Task 11; proper evaluation fits the vectorizer only on training data.

**12. No meaning.** The review vectors share `the`, `food`, and `was`, but `tasty`, `delicious`, and `terrible` are separate dimensions. Therefore, count-vector cosine similarity measures shared tokens rather than semantic similarity, and it does not know that `tasty` and `delicious` are synonyms.

### Vectorizer settings for a spam detector

I would compare word-level counts and TF-IDF using a stratified validation split, and test unigrams against `(1, 2)` n-grams. Unigrams can capture useful spam signals such as “free”, “win”, “cash”, and “prize”, while bigrams may capture phrases such as “claim now”. I would not blindly remove stopwords because negation and seemingly common words can matter; I would also avoid an aggressively low `max_features` or high `min_df`, since rare spam indicators may be removed. All vectorizer fitting and vocabulary selection must be done on training data only.

## References

1. Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval. _Journal of Documentation, 28_(1), 11–21. https://doi.org/10.1108/eb026526
2. Robertson, S. (2004). Understanding inverse document frequency: On theoretical arguments for IDF. _Journal of Documentation, 60_(5), 503–520. https://doi.org/10.1108/00220410410560573
3. scikit-learn developers. _Feature extraction: Text feature extraction_. https://scikit-learn.org/stable/modules/feature_extraction.html
4. scikit-learn developers. _TfidfVectorizer API reference_. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5. Ali, M. M. (2026). Resource-efficient Somali news classification using TF-IDF and linear SVM: A supervised baseline on a curated news corpus. _Scientific Reports_. https://doi.org/10.1038/s41598-026-68159-5
6. Abdirahman, A. A., Hashi, A. O., Dahir, U. M., Elmi, M. A., & Rodriguez, O. E. R. (2023). Enhancing natural language processing in Somali text classification: A comprehensive framework for stop word removal. _International Journal of Engineering Trends and Technology, 71_(12), 40–49. https://doi.org/10.14445/22315381/IJETT-V71I12P205
