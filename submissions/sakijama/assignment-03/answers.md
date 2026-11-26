# Assignment 3 — Vectorization

## Part A — Research

### 1. Text vectorization and one-hot encoding

Text vectorization converts linguistic data into numerical feature representations. Conventional classifiers, including linear classifiers, expect a fixed feature dimension: if feature position 2 means *free* for one message, it must mean *free* for every other message. This lets a model learn one coefficient per feature and process documents of different lengths in a shared mathematical space [1].

One-hot encoding assigns each vocabulary item a vector with one active position and zeros elsewhere. For example, a customer-service vocabulary `[refund, delivery, account]` encodes `delivery` as `[0, 1, 0]`. However, a five-word message produces five separate one-hot vectors while a ten-word message produces ten; that variable sequence is not itself one fixed-length document feature vector. Aggregation, such as Bag of Words, is needed for a standard document-level classifier. One-hot vectors also do not encode similarity between words [1, 2].

### 2. Bag of Words and the document-term matrix

Bag of Words (BoW) represents each document by the occurrence count of every term in a fixed vocabulary. Stacking the document vectors creates a **document-term matrix** with one row for each document, one column for each retained term, and a cell containing the term's count in that document. It is called a *bag* because token order is discarded while multiplicity is retained [1].

For a simple spam filter, merely detecting words associated with promotions may be useful even without their sequence. For sentiment classification, order can be essential: *I do not recommend this* and *I recommend this* have very different attitudes, yet treating words as independent counts fails to encode the phrase-level negation correctly. Short n-grams can partly address this limitation [1, 2].

### 3. Vocabulary size and sparsity

**Vocabulary size** is the number of unique feature terms learned by the vectorizer, not the number of token occurrences. A corpus can contain millions of word occurrences but far fewer distinct terms. As documents multiply, names, spelling variants, specialized terms, and inflections enlarge the feature space. Nevertheless, each document typically uses only a small fraction of the full vocabulary, creating a matrix dominated by zeros. **Sparse storage**, such as compressed sparse row (CSR), records nonzero values together with their column locations and row boundaries rather than allocating a dense value for each zero [1, 3].

| Choice | Effect on vocabulary |
|---|---|
| `min_df=2` | Keep a term only when it appears in at least two different documents. |
| `max_df=0.8` | Exclude terms appearing in more than 80% of documents; may remove ubiquitous words. |
| `max_features=5` | Retain up to five highest-frequency terms. |
| Lowercasing | Merge capitalization variants such as `WIN` and `win`. |
| Stopword removal | Exclude selected common terms, potentially removing useful negation. |
| Stemming | When applied consistently before vectorization, merge some inflected forms into shared features. |

These settings should be chosen using training data and validation results, because rare terms can be important in fraud or medical texts [3].

### 4. N-grams and word order

An **n-gram** is a sequence of *n* adjacent tokens: a **unigram** contains one token, a **bigram** two, and a **trigram** three. In an original example, the message *never arrived today* has unigrams `never`, `arrived`, `today`; bigrams `never arrived`, `arrived today`; and the trigram `never arrived today`. The bigram `never arrived` preserves a crucial negation that isolated counts cannot represent as a phrase [1].

Adding bigrams and trigrams creates additional features and thus a larger, sparser matrix. Rare phrases may also generalize poorly, and longer n-grams require more examples to estimate reliably. Therefore, n-gram order is a trade-off between local context and memory, speed, and data requirements [1, 3].

### 5. TF-IDF, its origin, and implementation differences

**Term Frequency–Inverse Document Frequency (TF-IDF)** gives a term more weight when it is frequent inside a particular document but uncommon across the corpus. Karen Spärck Jones's 1972 work on **term specificity** established the influential argument that collection-wide rarity should help determine a term's usefulness in information retrieval [4]. A standard introductory textbook version is:

- `TF(t,d) = count(t,d) / number_of_tokens(d)`
- `IDF(t) = ln(N / df(t))`
- `TF-IDF(t,d) = TF(t,d) × IDF(t)`

Here `N` is the number of documents and `df(t)` is the number containing term `t`. A distinctive term receives a larger IDF because its denominator is smaller. These are not the only conventions used in information retrieval [1, 4].

Scikit-learn's default `TfidfVectorizer` instead uses **raw count** for TF, smoothed IDF `ln((1+N)/(1+df(t))) + 1`, and then normalizes each document vector to **unit L2 length**. The defaults are `smooth_idf=True`, `norm='l2'` and `sublinear_tf=False`. This explains why its displayed values do not equal the textbook example, even if the ranking of terms agrees. TF-IDF values quantify representation weight, **not the probability** of a document belonging to a category [5].

### 6. Fitting, transforming, unknown words and leakage

`fit()` learns the vocabulary from training documents; for TF-IDF it also learns document-frequency statistics. `transform()` maps further documents into that **unchanged** feature space, while `fit_transform()` carries out learning and conversion on the same training collection. A test word absent from the training vocabulary is **out of vocabulary (OOV)** and has no separate feature column; a whole document can even become an all-zero vector [3, 5].

**Data leakage** occurs when information from held-out examples enters training-time decisions. Fitting a vectorizer on the entire dataset before splitting exposes test-only vocabulary, and TF-IDF fitting also exposes test-document frequency patterns. Such an evaluation does not simulate a genuinely unseen test set and can make reported performance overly optimistic, although the numerical effect varies with the experiment. The correct sequence is **split data → fit the vectorizer on training data → transform training and test using the fitted vectorizer → train on training labels → evaluate on held-out labels**. During model selection, use validation folds or a pipeline that fits preprocessing separately within each training fold [3, 5].

### 7. African-language case study: Amharic semantic representations

**Study:** Yimam, Ayele, Venkatesh, Gashaw, and Biemann (2021), *Introducing Various Semantic Models for Amharic: Experimentation and Evaluation with Multiple Tasks and Datasets* [6]. The language is **Amharic**, an Ethiopian language with relatively limited NLP resources. The authors developed or evaluated Amharic semantic representations and assessed them on tasks including sentiment classification, named entity recognition and word similarity. For sentiment classification, they used **positive, negative and neutral** classes and reported macro-averaged precision, recall and F1 on test data.

The sentiment experiments compared **TF-IDF features with conventional classifiers** (logistic regression, random forest, KNN, nearest centroid, and SVM) against **word embeddings** (AmWord2Vec and fastText), **contextual embeddings** (e.g. AmFlair and MultFlairFT), and graph-based representations. On the sentiment test, the paper reports macro F1 **52.92** for TF-IDF + logistic regression, **55.22** for AmWord2Vec, **56.92** for MultFlairFT and **58.48** for Role2Vec. These results show that several semantic representations performed better than that TF-IDF baseline in this setting; they do **not** prove that every embedding beats every TF-IDF model, since models and representations also differ [6].

BoW and TF-IDF do not inherently recognize that two different words have similar meanings: each unique word occupies its own independent column. Embedding approaches can learn related word representations from context, a motivation for the Amharic comparisons. Importantly, the study's semantic similarity experiments are additional evidence about captured meaning; sentiment F1 differences alone cannot isolate semantic understanding as the sole cause of a performance difference [6].

---

## Part B — Practical

**Reproducibility:** The reported figures below are derived from running `vectorization_practice.py` with NumPy and scikit-learn; the complete printed trace is saved in `output.txt`. The default `CountVectorizer` token pattern drops one-letter words and punctuation, and lowercases its vocabulary.

### Task 1 — One-hot encoding

The vocabulary is `[cash, class, free, lunch, notes, win]`. The script produces `free → [0,0,1,0,0,0]`, `cash → [1,0,0,0,0,0]`, and `win → [0,0,0,0,0,1]`. The phrase `free cash` produces two word vectors, whereas `win free cash` produces three, because one-hot encoding assigns a separate vector to every token rather than one fixed-length vector to the entire sentence.

### Task 2 — Bag of Words and document-term matrix

Rows D1–D5 represent the five documents; the alphabetical column names below match the script's learned vocabulary. The entry at **D1 × win** is **2** because `Win` and `WIN` are both lowercased to `win`. The transformed messages `win free phone` and `phone free win` have identical vectors, confirming that BoW loses word order.

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

### Task 3 — Tokens versus vocabulary

The five documents contain **41** tokens after `CountVectorizer`'s default tokenization and have **32** vocabulary entries. Words occurring in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today` and `win` (each appears in two documents). These counts use the vectorizer's analyzer, not naive whitespace splitting, to keep the report consistent with the matrix.

### Task 4 — Preprocessing inside the vectorizer

On D1, the default vectorizer lowercases both `Win` and `WIN` into the single feature `win`; with `lowercase=False` they become separate `Win` and `WIN` entries. The one-character token `a` and punctuation `!` do not enter the default vocabulary. With English stopwords, D1 loses `now` and `to`; on the complete corpus, `not` is also removed. Removing `not` is a risk for D5 because *not good* conveys a negative evaluation that the surviving word *good* alone does not.

### Task 5 — Vocabulary limits

The complete vocabulary has **32** words; `min_df=2` retains **7** (`free`, `lecture`, `notes`, `now`, `the`, `today`, `win`), and `max_features=5` retains **5** (`free`, `lecture`, `now`, `the`, `win`). The retained features are relatively common across the collection, whereas distinctive terms such as `cash` and `phone` can disappear. Strong frequency limits could therefore remove useful spam signals.

### Task 6 — N-grams

D5's bigram `not good` preserves the reviewer's negative opinion and `were great` helps describe the positive comment about notes. Adding features expands the full-corpus vocabulary from **32** unigrams to **67** unigrams/bigrams and **98** unigrams/bigrams/trigrams. The added phrases capture local word order but substantially increase feature dimensions.

### Task 7 — TF-IDF and D2 comparison

On D2, each of the eight vectorized words appears once, but the corpus-wide **document frequency** differs. `call`, `cash`, `entry` and `prizes` each appear in one document and receive higher weights than `free`, `now`, `today` and `win`, each found in two. On D1, `win` occurs twice, and its resulting TF-IDF weight (**0.5739**) exceeds every other term's weight even though its IDF is lower than that of D1-only words.

**D2 count versus TF-IDF (rounded to four decimals):**

| Word | Count | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.3891 |
| cash | 1 | 0.3891 |
| entry | 1 | 0.3891 |
| free | 1 | 0.3140 |
| now | 1 | 0.3140 |
| prizes | 1 | 0.3891 |
| today | 1 | 0.3140 |
| win | 1 | 0.3140 |

### Task 8 — Textbook TF-IDF by hand

The corpus has `N = 5` documents; D2 contains **8 tokens** under the vectorizer's tokenizer (single-letter items and punctuation excluded). Both `cash` and `today` appear once, so their normalized term frequencies are both `TF = 1/8 = 0.125`.

| Word | count | df | TF | IDF = ln(N/df) | Textbook TF-IDF | scikit-learn TF-IDF |
|---|---:|---:|---:|---:|---:|---:|
| cash | 1 | 1 | 0.125000 | 1.609438 | 0.201180 | 0.389141 |
| today | 1 | 2 | 0.125000 | 0.916291 | 0.114536 | 0.313957 |

Thus `cash` ranks above `today` by both methods. The magnitudes differ because scikit-learn uses raw counts, smoothed IDF with `+1`, and L2 normalization by default, rather than the chosen textbook convention [5].

### Task 9 — Sparse matrix

The count matrix contains **160** cells (`5 × 32`), of which **39** are nonzero and **121** are zero, giving **75.62% zeros**. Scikit-learn returns a CSR sparse matrix, so the many zero cells need not be stored as ordinary dense entries. This is particularly useful with large vocabularies.

### Task 10 — Fit on D1–D4 and transform test messages

After fitting on D1–D4, N1's unknown terms are **`prize` and `your`**, while N2's are **`is`, `moved` and `tomorrow`**. The vectorizer recognizes `prizes` from D2 but does not automatically equate that plural form with singular `prize`. A consistent **lemmatization** step (`prizes → prize`), or an appropriate stemming step, before fitting and transforming could resolve that mismatch; the same preprocessing must be applied on both sides.

### Task 11 — Data leakage

When the vectorizer is fitted on D1–D4 together with N1 and N2, test-only words **`is`, `moved`, `prize`, `tomorrow` and `your`** enter the learned vocabulary. They are absent from the training-only vocabulary, demonstrating leakage from unseen test text into the feature-building stage. Vectorizer fitting must therefore use training data only.

### Task 12 — No meaning: cosine similarity

The three review vectors share `the`, `food` and `was` but have distinct features for `tasty`, `delicious` and `terrible`. Cosine similarity is **0.75** for every distinct pair: R1–R2, R1–R3 and R2–R3. Thus R1 is **not** more similar to the semantically positive R2 than to the negative R3 under these count vectors; lexical overlap alone does not recognize synonymy or opposition.

| Cosine similarity | R1 tasty | R2 delicious | R3 terrible |
|---|---:|---:|---:|
| R1 tasty | 1.00 | 0.75 | 0.75 |
| R2 delicious | 0.75 | 1.00 | 0.75 |
| R3 terrible | 0.75 | 0.75 | 1.00 |

### Recommended vectorization choices for a spam detector

I would compare **CountVectorizer and TfidfVectorizer** using validation data rather than assume either is always better. The script shows useful spam-related features such as `win`, `cash`, `phone`, `free` and `claim`; TF-IDF distinguishes corpus-specific terms while counts retain repeated emphasis (`win` occurs twice in D1). I would try **unigrams plus bigrams**, retaining phrases that might indicate offers or requests, but avoid aggressively removing stopwords or setting `min_df`/`max_features` so tightly that useful infrequent terms like `cash` vanish. I would also preserve or separately encode relevant monetary amounts, phone patterns and capitalization where appropriate, and fit all preprocessing and vectorization rules on training data only.

---

## References

[1] Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://doi.org/10.1017/CBO9780511809071

[2] Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (online draft). https://web.stanford.edu/~jurafsky/slp3/

[3] Scikit-learn developers. *CountVectorizer: Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

[4] Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526

[5] Scikit-learn developers. *TfidfTransformer and TfidfVectorizer: Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html and https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[6] Yimam, S. M., Ayele, A. A., Venkatesh, G., Gashaw, I., & Biemann, C. (2021). Introducing various semantic models for Amharic: Experimentation and evaluation with multiple tasks and datasets. *Future Internet, 13*(11), 275. https://doi.org/10.3390/fi13110275

