# Assignment 3 – Vectorization

Sagal Mohamed Yusuf

*From words to numbers: one-hot encoding, Bag of Words, n-grams, TF-IDF, fit/transform, data leakage, and the limits of counting*

# Part A: Research

## 1. Text vectorization, fixed-length features and one-hot encoding

### 1.1 Definition

Text vectorization is the process of mapping a piece of text (a word, sentence or whole document) to a vector of numbers so that an algorithm can process it. It is the first step of most classical NLP pipelines: raw text → tokens → vocabulary → numeric vectors → classifier (Jurafsky & Martin, 2024; Manning, Raghavan & Schütze, 2008).

### 1.2 Why every document needs the same number of features

Most ML models (logistic regression, SVMs, decision trees, naive Bayes, standard neural networks) learn a function from a fixed-length input vector to a label. They are defined by parameters attached to specific feature positions: a logistic regression has one weight per column, and column *j* must mean the same thing for every row. Documents, however, differ in length: a tweet has 15 words and a news article has 800. If we fed the raw word sequence, the number of inputs would change from one example to the next, and there would be no consistent column to attach a weight to. Vectorization solves this by choosing one shared feature space (typically one dimension per vocabulary item) and describing every document inside it, so every document becomes a row of the same width, even if most cells are zero. (Sequence models such as RNNs and Transformers handle variable length differently, but they too first convert each token into a fixed-size vector.)

### 1.3 One-hot encoding of words

In one-hot encoding, the vocabulary is fixed at size *V*, each word is given an index, and a word is represented by a vector of length *V* with a single 1 at its index and 0 everywhere else.

**Real-life example.** Suppose an online shop has only six distinct words across its customer reviews: *delivery, fast, refund, slow, the, was*. Then:

| Word | delivery | fast | refund | slow | the | was |
|---|---|---|---|---|---|---|
| fast | 0 | 1 | 0 | 0 | 0 | 0 |
| slow | 0 | 0 | 0 | 1 | 0 | 0 |
| refund | 0 | 0 | 1 | 0 | 0 | 0 |

### 1.4 Why one-hot vectors are not enough for a document

- **A document is many words, not one.** The review "the delivery was fast" becomes four one-hot vectors (a 4 × 6 matrix); a ten-word review becomes a 10 × 6 matrix. The shape still varies with document length, so the fixed-width requirement is not met. Some aggregation (sum, max, average) is needed, and that step is exactly what BoW does.
- **No notion of similarity.** Any two different one-hot vectors are orthogonal (dot product 0). "fast" is as far from "quick" as it is from "refund" (Jurafsky & Martin, 2024).
- **Very high dimension.** The length equals the vocabulary size, which in real corpora is tens or hundreds of thousands.
- **No order or context.** Nothing in the representation says which word came before which.

## 2. Bag of Words and the document-term matrix

### 2.1 What the matrix holds

In the Bag of Words model each document is represented by the counts (or presence/absence) of the words it contains, ignoring grammar and order. Collecting all documents gives the **document-term matrix (DTM)**:

- **Rows** = documents (one per review, email, article).
- **Columns** = terms in the vocabulary (the features).
- **Cell (i, j)** = the weight of term *j* in document *i*: usually the raw count, sometimes a binary 0/1, or a TF-IDF weight.

The transposed layout (terms as rows) is called the term-document matrix; scikit-learn uses the document-term layout so rows are samples.

A hand-built example with three tiny reviews: d1 = "the delivery was fast", d2 = "the delivery was slow", d3 = "fast delivery, fast refund".

| Document | delivery | fast | refund | slow | the | was |
|---|---|---|---|---|---|---|
| d1 | 1 | 1 | 0 | 0 | 1 | 1 |
| d2 | 1 | 0 | 0 | 1 | 1 | 1 |
| d3 | 1 | 2 | 1 | 0 | 0 | 0 |

Note that "fast" appears twice in d3, so its cell holds 2. Each row has the same length (6), which solves the fixed-width problem from Section 1.

### 2.2 Why it is called a "bag"

A bag (multiset) is a collection in which membership and multiplicity matter but order does not. Putting the words of a document "in a bag" and shaking it destroys their sequence while preserving how many times each appears. The idea goes back to early statistical text retrieval, in which word frequencies in a document were taken as evidence of its content (Luhn, 1957), and it is related to the distributional view that a word's behaviour is characterised by the contexts it occurs in (Harris, 1954).

### 2.3 When losing word order matters little, and when it hurts

- **Little harm: topic classification or spam filtering.** Whether a news article is about sport or economics is largely decided by which words occur ("match", "goal", "league" versus "inflation", "budget"). Word order adds little. Zhang, Zhao & LeCun (2015) found that even simple BoW and TF-IDF with a linear classifier were strong baselines on topic-style datasets.
- **Real problems: sentiment with negation, and meaning that depends on who did what.** "The service was not good" and "The service was good, not slow" contain nearly the same bag of words but express opposite attitudes, and "dog bites man" and "man bites dog" have identical bags. Negation, contrast and role assignment are carried by order, so a pure BoW cannot represent them.

## 3. Vocabulary size and sparsity

### 3.1 Why matrices become huge and mostly zero

The vocabulary is the set of distinct tokens in the training corpus. Natural language has a long tail (a few words are very frequent, most words are rare, an observation often associated with Zipf's law; Manning et al., 2008), so as a corpus grows, new rare words (names, misspellings, numbers, inflected forms) keep appearing. The number of columns therefore grows with the corpus. But any single document uses only a tiny fraction of the vocabulary, so each row contains mostly zeros.

**Illustration (assumed numbers).** Take 100,000 documents and a 200,000-word vocabulary. A dense matrix of 8-byte floats needs 100,000 × 200,000 × 8 bytes = 1.6 × 10¹¹ bytes, about 160 GB. If each document has on average 150 distinct words, only 100,000 × 150 = 15 million cells are non-zero (0.075%).

### 3.2 Sparse matrix storage

Sparse formats store only the non-zero entries. scikit-learn's vectorizers return SciPy sparse matrices (Virtanen et al., 2020; Pedregosa et al., 2011). In the common CSR (Compressed Sparse Row) format three arrays are kept: `data` (the non-zero values), `indices` (the column index of each value) and `indptr` (where each row starts inside those arrays). In the illustration above, 15 million non-zeros need about 15M × (8 bytes value + 4 bytes column index) ≈ 180 MB plus a small row-pointer array, instead of 160 GB. Other formats (COO, CSC) suit construction or column slicing. Converting to a dense array (`.toarray()`) on a large corpus can exhaust memory and should be avoided.

### 3.3 Controlling the vocabulary

| Setting | What it does | Effect on vocabulary |
|---|---|---|
| `min_df` | Ignore terms appearing in fewer than this many documents (integer count) or fraction of documents (float) | Removes very rare terms (typos, one-off names); shrinks vocabulary substantially and reduces noise |
| `max_df` | Ignore terms appearing in more than this count or fraction of documents | Removes near-ubiquitous words that carry little discriminating information (a corpus-specific stopword filter) |
| `max_features` | Keep only the top *k* terms ordered by term frequency across the corpus | Hard cap on dimensionality; keeps frequent terms, may discard informative rare ones |
| Lowercasing | Maps "Delivery" and "delivery" to one token | Merges variants, reduces size; may lose distinctions (e.g., "US" vs "us") |
| Stopword removal | Drops function words ("the", "was") from a list | Smaller vocabulary; but removing "not" can destroy negation, and generic English lists may not suit other languages or domains |
| Stemming / lemmatization | Reduces words to a stem (Porter, 1980) or dictionary form | Merges inflected forms ("delivered", "delivering" → "deliver"), shrinking vocabulary; stems can be crude and merge unrelated words |

In scikit-learn, `CountVectorizer` and `TfidfVectorizer` apply lowercasing by default, use a default token pattern that keeps tokens of two or more word characters (so single-character tokens are dropped), and apply no stopword removal or stemming unless requested (scikit-learn developers, 2024). Because every one of these choices changes which columns exist, they are hyper-parameters of the whole pipeline and should be chosen using training or validation data only.

## 4. N-grams

An n-gram is a contiguous sequence of *n* tokens. Instead of using single words as features, we can also use pairs or triples.

- **Unigrams** (n = 1): single words: "not", "good".
- **Bigrams** (n = 2): "not good", "very slow".
- **Trigrams** (n = 3): "not very good", "cash on delivery".

Because an n-gram is a sequence, a bag of n-grams retains **local** word order (within a window of *n* words) even though global order is still lost (Manning & Schütze, 1999; Jurafsky & Martin, 2024). In scikit-learn this is controlled by `ngram_range=(1, 2)` and similar.

**Example where a bigram changes the meaning.** Take two reviews: "the phone is good" and "the phone is not good". With unigrams only, both contain "good", pushing a sentiment classifier toward the positive class for both. The bigram "not good" is a separate feature that a classifier can learn is negative. Similarly, "New York" or "machine learning" mean something different from their separate words, and in Somali-language work the same applies to any multiword expression whose meaning is not the sum of its parts.

**Costs of larger n.**

- **Feature explosion.** The number of possible n-grams grows roughly as V^n; the observed number grows more slowly but still far faster than the unigram vocabulary. For example, Zhang et al. (2015) limited their baselines to 50,000 words but needed to select the 500,000 most frequent n-grams (up to 5-grams).
- **Greater sparsity.** Most long n-grams appear once or never, so few of them are shared across documents and models tend to overfit.
- **Memory and time.** More columns means larger matrices and slower training.
- **Diminishing returns.** Local order is captured, but long-range dependencies and meaning still are not.

In practice bigrams (sometimes trigrams) combined with a `min_df` threshold are a common compromise.

## 5. TF-IDF

### 5.1 Idea and origin

Raw counts treat all words equally, so frequent function words ("the", "was") dominate even though they say little about a document. TF-IDF multiplies a term's frequency in a document (TF) by a factor that reduces the weight of terms occurring in many documents (IDF). The IDF idea was introduced by Karen Spärck Jones (1972) in a paper on term specificity: a term that occurs in few documents is more specific and therefore more useful for discriminating between documents, so its weight should be inversely related to the number of documents containing it. The combination of term frequency with inverse document frequency was systematised and compared in the SMART retrieval experiments (Salton & Buckley, 1988). Robertson (2004) later reviewed theoretical arguments for why IDF works, including probabilistic and information-theoretic justifications. IDF began as a heuristic that proved empirically robust, and theory came afterwards.

### 5.2 Textbook formula

For term *t*, document *d* and a corpus of *N* documents:

- tf-idf(t, d) = tf(t, d) × idf(t)
- idf(t) = log( N / df(t) )

where tf(t, d) is the count (or a normalised count) of *t* in *d* and df(t) is the number of documents containing *t* (Manning et al., 2008). A term in every document has df = N, so idf = log(1) = 0 and its weight vanishes. A term in one document has the largest idf, log(N). Hence **rarer words receive higher weights**: IDF is a decreasing function of document frequency.

### 5.3 How scikit-learn's TfidfVectorizer differs

According to the scikit-learn documentation (scikit-learn developers, 2024), with default settings:

- **Smoothed IDF with an additive 1:** idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1. Smoothing acts as if one extra document containing every term had been seen, which prevents division by zero for unseen terms. The "+1" means terms present in every document are not completely ignored (idf = 1 instead of 0). Setting `smooth_idf=False` gives ln(N/df) + 1.
- **L2 normalisation:** each document vector is rescaled to unit Euclidean length (`norm='l2'`), so long and short documents become comparable and the dot product of two rows equals their cosine similarity.
- **Raw counts for TF** by default, with optional `sublinear_tf=True` to use 1 + ln(tf).
- The natural logarithm is used.

**Worked example** on the corpus from Section 2 (N = 3):

| Term | df | Textbook ln(N/df) | sklearn ln((1+N)/(1+df)) + 1 |
|---|---|---|---|
| delivery | 3 | 0.000 | 1.000 |
| fast | 2 | 0.405 | 1.288 |
| refund | 1 | 1.099 | 1.693 |
| slow | 1 | 1.099 | 1.693 |
| the | 2 | 0.405 | 1.288 |
| was | 2 | 0.405 | 1.288 |

For d3 (counts: delivery 1, fast 2, refund 1), the unnormalised sklearn weights are 1 × 1.000 = 1.000, 2 × 1.288 = 2.575 and 1 × 1.693 = 1.693. The L2 norm is √(1.000² + 2.575² + 1.693²) ≈ 3.240, so the final vector is delivery ≈ 0.309, fast ≈ 0.795, refund ≈ 0.523. Under the textbook formula "delivery" would get weight 0; under scikit-learn it keeps a small positive weight. (You can verify these numbers by running `TfidfVectorizer` on the three sentences and reading `idf_`.)

### 5.4 TF-IDF values are weights, not probabilities

A probability must lie in [0, 1] and the probabilities over outcomes must sum to 1. TF-IDF values do not follow these rules: before normalisation they are products of a count and a log-based factor and can be greater than 1; after L2 normalisation it is the *squares* of the entries that sum to 1, not the entries themselves. They do not estimate P(word | document) or P(document | word). They are heuristic importance scores that say how characteristic a term is of a document relative to the corpus, and a high value means "frequent here and rare elsewhere", not "likely".

## 6. fit, transform and fit_transform; out-of-vocabulary words; data leakage

### 6.1 The three methods

- **`fit(X_train)`** learns parameters from data. For `CountVectorizer`, that is the vocabulary (the mapping term → column index); for `TfidfVectorizer`, additionally the IDF values (`idf_`), which depend on document frequencies computed over the fitted documents.
- **`transform(X)`** applies what was learned to any set of documents, producing the matrix, without learning anything new.
- **`fit_transform(X_train)`** does both on the same data in one call (and can be more efficient). Use it on training data only.

### 6.2 Out-of-vocabulary (OOV) words

A word in new text that is not in the fitted vocabulary has no column, so `transform` silently ignores it. A test document made up entirely of unseen words becomes an all-zero row. This is correct behaviour: the model cannot have learned a weight for a feature it never saw. The weakness is that unseen words that may be informative are simply lost, which is a major issue for low-resource and morphologically rich languages where many surface forms are rare. Alternatives include hashing vectorizers, subword/character n-grams or learned embeddings.

### 6.3 Data leakage in vectorization

Data leakage is when information from outside the training set (especially from the test set) influences the training process, so the measured performance does not reflect performance on genuinely new data (Kaufman et al., 2012; scikit-learn developers, 2024, "Common pitfalls").

The typical mistake is to call `fit_transform` on the **entire** dataset and split afterwards. Then:

- the vocabulary includes words that occur only in the test documents, and `min_df`/`max_df`/`max_features` decisions are made partly using test documents;
- the IDF values encode document frequencies computed with test documents;
- cross-validation repeats the problem: in every fold the validation documents helped fit the vectorizer.

The model is therefore evaluated with a feature space that has already "seen" the test text. Results become optimistic because the test set is no longer fully unseen and the evaluation no longer mimics deployment, where new documents arrive after the vectorizer is frozen. For an unsupervised step like TF-IDF the inflation is often modest, but it can be larger with small datasets, aggressive feature selection or when label information is used (for instance chi-square selection of terms), and it is a methodological error either way. Don't assume it is harmless; avoid it.

**Correct order of steps:**

1. Split the labelled data into train and test (and validation, if used) **first**.
2. Fit the vectorizer on the training text only (`fit_transform(X_train)`).
3. Train the classifier on the resulting training matrix.
4. Apply the already fitted vectorizer to the test text with `transform(X_test)` (never `fit`).
5. Evaluate on the test matrix.
6. For cross-validation, wrap vectorizer and classifier in a scikit-learn `Pipeline` so the vectorizer is re-fitted inside each training fold automatically.

## 7. Case study: comparing text representations for news classification in isiZulu and Siswati

**Why this study.** I searched for a Somali study that directly compares BoW, TF-IDF and embeddings. I found Somali work on sentiment analysis with classical ML and deep learning (Abdirahman et al., 2023) and a Somali language model, SomBERTa (arXiv:2503.18117), but I could not verify from the material I accessed that they compare these exact representations, so I did not use them as the main case. The study below does make that comparison, for two African languages, and its tables are fully available.

**Reference:** Madodonga, A., Marivate, V., & Adendorff, M. (2023). *Izindaba-Tindzaba: Machine learning news categorisation for long and short text for isiZulu and Siswati.* arXiv:2306.07426.

### 7.1 Task

News topic classification: assign each news item to a topic category. Articles were first labelled with IPTC news categories (17 categories), and categories with too few examples were removed, leaving five categories per language.

### 7.2 Data

- isiZulu: 752 items (full articles and titles) from the Isolezwe newspaper website; 563 remained after small categories were removed.
- Siswati: 80 news headlines from a public-broadcaster radio Facebook page; 68 remained.
- Three experimental datasets: isiZulu articles (long text), isiZulu titles (short text) and Siswati titles (short text).
- Vectorizers and embeddings were built on larger unlabelled corpora (about 5.07 million isiZulu tokens and 0.53 million Siswati tokens from SADILAR and the Leipzig corpora).
- Because the labelled data were small and imbalanced, the authors also used contextual data augmentation (replacing words by similar words found with Word2vec) and SMOTE oversampling.

### 7.3 Representations and models compared

BoW and TF-IDF (each capped at 20,000 tokens) and 300-dimensional Word2vec (Mikolov et al., 2013), combined with logistic regression, naive Bayes and XGBoost, plus an LSTM on Word2vec. Evaluation used 5-fold cross-validation and no hyper-parameter tuning, so the authors describe the results as baselines.

### 7.4 Results (F1-score, %)

| Dataset and setting | Best BoW | Best TF-IDF | Best Word2vec |
|---|---|---|---|
| isiZulu articles, original | 36.06 (logistic regression) | 29.34 (XGBoost) | 82.78 (LSTM); 65.13 (logistic regression) |
| isiZulu titles, original | 17.38 (logistic regression) | 17.58 (logistic regression) | 72.01 (LSTM); 39.18 (logistic regression) |
| Siswati titles, original | 26.34 (logistic regression) | 25.50 (XGBoost) | 81.06 (LSTM); 31.57 (naive Bayes) |
| isiZulu articles, augmented | 83.09 (logistic regression) | 79.92 (XGBoost) | 95.21 (XGBoost) |

The authors report that classifiers on Word2vec features outperformed those on TF-IDF and BoW across the datasets. On the original small datasets the gap is large: for example, isiZulu articles with logistic regression reach 65.13 F1 with Word2vec but 24.19 with TF-IDF. After augmentation the gap narrows, because adding examples helps the sparse representations more, but Word2vec still wins. Long articles gave better results than short titles. The authors state that the poor performance of TF-IDF and BoW remains something to be investigated.

### 7.5 What this shows about the limitation of BoW and TF-IDF

BoW and TF-IDF give every word its own independent column, so "police" and "officers" or two inflected forms of the same isiZulu word are unrelated features. A classifier trained on 563 or 68 examples can only learn from words it has literally seen in training, whereas Word2vec places related words near each other in a dense space learned from millions of unlabelled tokens, so knowledge transfers between related words. This is the general limitation described in the survey literature: TF-IDF cannot account for the similarity between words because each word is indexed independently (Kowsari et al., 2019). The Izindaba-Tindzaba results are consistent with that explanation, especially in the low-data setting.

**Caveats I would note in a critical reading:**

- The paper reports the pattern but does not run an experiment isolating *meaning* as the cause; the authors leave the explanation for future work. The "no shared meaning between columns" explanation comes from the general literature, not from a test in this paper.
- Word2vec was trained on a much larger external corpus than the labelled data, so part of its advantage may come from extra unlabelled text rather than from "meaning" as such; BoW and TF-IDF could not use that text for supervision.
- isiZulu and Siswati are agglutinative languages with many surface word forms, which probably makes BoW/TF-IDF matrices sparser. This is a plausible additional factor, but the paper does not test it.
- The paper does not state clearly whether augmentation was applied only to training folds. If augmented copies of the same sentence fall into both training and validation folds, the high augmented scores could be inflated, which is the same kind of leakage discussed in Section 6.
- The datasets are very small (68 Siswati items), so confidence intervals are wide (the paper reports them) and the results are baselines, not definitive rankings.

### 7.6 A balanced conclusion

BoW and TF-IDF remain useful: they are fast, interpretable and strong on large topic-classification datasets (Zhang et al., 2015). But they treat words as unrelated symbols, they cannot see order beyond n-grams, and they have no representation for unseen words. When labelled data are scarce, as in many African languages, representations that carry information about word similarity (embeddings, or pretrained language models) have a clear advantage, as this case study illustrates.

# Part B: Practical

All numbers below come from `vectorization_practice.py` (saved output in `output.txt`).

## Document-term matrix (Task 2)

CountVectorizer fitted on D1-D5: 5 documents x 32 words (shape (5, 32)). Rows are documents, columns are words, cells are counts.

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

## Count vs TF-IDF for every word in D2 (Task 7)

D2: "Free entry: win cash prizes today. Call now." (TfidfVectorizer fitted on D1-D5, N = 5; rows are L2-normalised.)

| Word | Count in D2 | df | IDF (sklearn) | TF-IDF (sklearn) |
|---|---|---|---|---|
| call | 1 | 1 | 2.099 | 0.389 |
| cash | 1 | 1 | 2.099 | 0.389 |
| entry | 1 | 1 | 2.099 | 0.389 |
| free | 1 | 2 | 1.693 | 0.314 |
| now | 1 | 2 | 1.693 | 0.314 |
| prizes | 1 | 1 | 2.099 | 0.389 |
| today | 1 | 2 | 1.693 | 0.314 |
| win | 1 | 2 | 1.693 | 0.314 |

## Hand calculations for D2 (Task 8)

Textbook formulas: TF = count / total tokens in D2, IDF = ln(N / df), TF-IDF = TF x IDF. D2 has 8 tokens (free, entry, win, cash, prizes, today, call, now) and N = 5 documents.

**cash:** count = 1, so TF = 1/8 = 0.1250. df = 1 (only D2), so IDF = ln(5/1) = 1.6094. TF-IDF = 0.1250 x 1.6094 = **0.2012**.

**today:** count = 1, so TF = 1/8 = 0.1250. df = 2 (D2 and D3), so IDF = ln(5/2) = 0.9163. TF-IDF = 0.1250 x 0.9163 = **0.1145**.

scikit-learn gave cash = 0.3891 and today = 0.3140 for D2.

## Short answers

**Task 1 - One-hot encoding.** The vector for `free` is [0, 0, 1, 0, 0, 0] and for `cash` it is [1, 0, 0, 0, 0, 0]; `win free cash` gives three such vectors. Every word gets its own vector, so a sentence of 2 words gives 2 vectors and a sentence of 3 words gives 3. The number of rows therefore changes with sentence length, so one-hot encoding does not give each document the same number of features.

**Task 2 - Bag of Words.** The cell with value 2 in D1 is the column `win`. D1 contains both `WIN` and `Win`, and CountVectorizer lowercases by default, so they are counted as the same word. The vectors for `win free phone` and `phone free win` are identical, because Bag of Words records which words occur and how often but not their order.

**Task 4 - Preprocessing.** The word `a` is missing from the default vocabulary because the default token pattern keeps only tokens of two or more characters, and `!` is dropped because punctuation is not part of a token. With `lowercase=False`, `WIN` and `Win` become two separate words, while the default merges them into `win`. The stopword list removed `now` and `to` from D1. On D1-D5 it also removed `not`, so D5 becomes `lecture good notes great`, which no longer shows that the reviewer thought the lecture was not good.

**Task 5 - Limiting the vocabulary.** The full vocabulary has 32 words, `min_df=2` leaves 7 (free, lecture, notes, now, the, today, win) and `max_features=5` leaves 5 (free, lecture, now, the, win). The words that survive are the ones that occur in several documents or most often. Words that occur only once, such as `phone`, `cash` and `prizes`, were removed, even though some of them look informative.

**Task 6 - N-grams.** The bigram `not good` keeps the reviewer's opinion, because unigrams alone give `not` and `good` as unrelated features. On D1-D5 the vocabulary grows from 32 features for (1, 1) to 67 for (1, 2) and 98 for (1, 3), about 2x and 3x the unigram size. Almost all of the new features are unique to one document (65 of 66 bigrams and trigrams occur in only one), so larger n-grams make the matrix wider and sparser.

**Task 7 - TF-IDF.** In D2 every word has count 1, so the differences come only from IDF: `call`, `cash`, `entry` and `prizes` occur in one document (IDF 2.099) and get 0.389, while `free`, `now`, `today` and `win` occur in two documents (IDF 1.693) and get 0.314. In D1 `win` gets the highest weight (0.574) because it occurs twice in D1 (count 2) while sharing the same IDF as `free` and `now`. The weights are not probabilities: they are importance scores, and each row is scaled to length 1.

**Task 8 - TF-IDF by hand.** My textbook calculation gives cash = 0.2012 and today = 0.1145, while scikit-learn gives 0.3891 and 0.3140. The numbers differ because scikit-learn uses the smoothed IDF ln((1+N)/(1+df)) + 1 and then L2-normalises the row, whereas my calculation uses ln(N/df) and relative term frequency. The ranking is the same in both: `cash` is higher than `today`, because `cash` is rarer (df = 1 against df = 2).

**Task 10 - Fit and transform.** The vectorizer was fitted on D1-D4, and the out-of-vocabulary words are `your` and `prize` in N1, and `is`, `moved` and `tomorrow` in N2; they are ignored and get no column. `prize` is unknown because D2 contains only `prizes`, and the vectorizer matches exact strings, so the two are different tokens. Stemming or lemmatization from Assignment 2 would reduce both words to one form and fix this.

**Task 11 - Data leakage.** The vocabulary fitted on D1-D4 has 26 words, and the one fitted on D1-D4 plus N1 and N2 has 31. The five words that leaked from the test set are `is`, `moved`, `prize`, `tomorrow` and `your`. This is leakage because the vocabulary (and, for TF-IDF, the IDF values) was influenced by test texts, so the test set no longer looks like truly new data and results can look better than they are. The correct order is to split first, fit the vectorizer on the training text only, and then only transform the test text.

**Task 12 - No meaning.** No: the cosine similarity of R1 and R2 (`tasty` vs `delicious`) is 0.750, exactly the same as R1 and R3 (`tasty` vs `terrible`), and every pair of reviews scores 0.750. Each pair shares three of four words (`the`, `food`, `was`), and `tasty`, `delicious` and `terrible` are separate, unrelated columns. Bag of Words can see that the words are different, but it cannot see that two of them mean nearly the same thing and the third means the opposite.

## Settings for a spam detector

For a spam detector I would use TF-IDF (or plain counts) with lowercasing, because in my output `WIN` and `Win` merged into one `win` with count 2, and repetition of words like this is a useful spam signal, while TF-IDF (D1: `win` = 0.574) also gives rarer words more weight than common ones like `the`. I would add bigrams, `ngram_range=(1, 2)`, because phrases such as `call now` or `claim prize` carry more than the single words, but I would avoid trigrams and larger, since 65 of 66 bigrams and trigrams in my corpus occurred in only one document and just made the matrix sparser. I would avoid the built-in English stopword list, because it removed `now` and `to` from D1, and `now` is an urgency word that occurs in both spam messages (D1 and D2), and it removed `not` from D5, which shows it can delete words that change meaning. I would also avoid aggressive vocabulary limits on a small corpus: `max_features=5` kept `the` and `lecture` but dropped `cash`, `prizes` and `claim`, and `min_df=2` reduced 32 words to 7 and dropped the same spam words; on a large real corpus a small `min_df` (such as 2 to 5) is reasonable to remove typos and one-off words. Finally, I would stem or lemmatize so that `prize` and `prizes` are the same feature (the Task 10 out-of-vocabulary problem), and fit the vectorizer on the training messages only to avoid the leakage seen in Task 11. With only five training documents these choices are based on tiny evidence, so on a real spam dataset I would confirm them with cross-validation.

## References

Abdirahman, A. A., Hashi, A. O., Dahir, U. M., Elmi, M. A., & Rodriguez, O. E. R. (2023). Comparative analysis of machine learning and deep learning models for sentiment analysis in Somali language. *SSRG International Journal of Electrical and Electronics Engineering, 10*(7). https://doi.org/10.14445/23488379/IJEEE-V10I7P104 (consulted as a possible Somali case; not used as the main case study).

Harris, Z. S. (1954). Distributional structure. *Word, 10*(2–3), 146–162.

Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed., draft). Chapters on vector semantics and embeddings. https://web.stanford.edu/~jurafsky/slp3/

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15.

Kowsari, K., Jafari Meimandi, K., Heidarysafa, M., Mendu, S., Barnes, L., & Brown, D. (2019). Text classification algorithms: A survey. *Information, 10*(4), 150. arXiv:1904.08067.

Luhn, H. P. (1957). A statistical approach to mechanized encoding and searching of literary information. *IBM Journal of Research and Development, 1*(4), 309–317.

Madodonga, A., Marivate, V., & Adendorff, M. (2023). Izindaba-Tindzaba: Machine learning news categorisation for long and short text for isiZulu and Siswati. arXiv:2306.07426.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. (Chapters 2 and 6.)

Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems, 26*.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.

Robertson, S. (2004). Understanding inverse document frequency: On theoretical arguments for IDF. *Journal of Documentation, 60*(5), 503–520.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.

scikit-learn developers. (2024). *scikit-learn user guide and API documentation*: "Text feature extraction", `CountVectorizer`, `TfidfVectorizer`, "Common pitfalls and recommended practices" (data leakage), `Pipeline`. https://scikit-learn.org

Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21.

Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*, 261–272.

Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *Advances in Neural Information Processing Systems, 28*. arXiv:1509.01626.
