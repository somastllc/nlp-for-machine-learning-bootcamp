# Assignment 3 – Vectorization

## Part A – Research

### 1. Text vectorization and one-hot encoding

Text vectorization is the process of converting textual documents into numerical feature representations that can be processed by machine-learning algorithms. A text collection is represented using a shared feature space: each position has a defined meaning (for example, a vocabulary term), so every document is represented by the same number of features. This fixed dimensionality is important because standard machine-learning estimators operate on rectangular feature matrices in which every row has the same number of columns.

In word-level one-hot encoding, each vocabulary item is represented by a binary vector. The vector contains a `1` at the position assigned to that word and `0` at all other positions. For example, with the vocabulary `[cat, dog, bird]`, `dog` becomes `[0, 1, 0]`. This representation identifies which word is present, but encoding a complete sentence this way produces a sequence of word vectors rather than one fixed-length document vector. A document-level method such as Bag of Words is therefore needed to combine word information into one representation.

A real-life example is a customer-support system that classifies incoming messages into categories such as billing, delivery, or technical support. The words in a message can first be converted to numerical features, after which a classifier can learn patterns associated with each category.

### 2. Bag of Words and the document-term matrix

Bag of Words (BoW) represents a document by recording the occurrence or frequency of each vocabulary term. After the same vocabulary has been applied to a collection of documents, the resulting document-term matrix (DTM) has one row per document and one column per vocabulary term. A cell contains the count of that term in that document (or, in binary variants, simply whether it occurs).

It is called a “bag” because the basic representation does not preserve the original order or grammatical relationships between words. For example, `free prize` and `prize free` receive the same count vector. Losing word order may do little harm in a simple spam-filtering task, where the presence of words such as “cash” or “prize” can be useful clues. It can cause serious problems in sentiment or intent analysis: `I like this` and `I do not like this` differ in meaning, and a representation that ignores the scope of “not” can miss that difference. N-grams can preserve some local word order.

### 3. Vocabulary size and sparsity

The vocabulary is the set of distinct terms retained by a vectorizer, and vocabulary size is the number of terms (therefore, the number of columns) in the feature matrix. A large real-world collection can contain many documents and many distinct words, while each individual document uses only a small fraction of the total vocabulary. Consequently, most document-term matrix cells are zero; this is called sparsity.

Sparse matrix formats avoid storing every zero explicitly. Formats such as compressed sparse row (CSR) store non-zero values together with index information describing their positions. This reduces memory use and can make sparse operations more efficient.

Vocabulary settings control which terms are retained. `min_df` removes terms that occur in fewer than a specified number or proportion of documents; `max_df` removes terms that occur in more than a specified number or proportion of documents; and `max_features` caps the vocabulary at a selected number of terms, usually choosing frequent terms. Lowercasing merges case variants such as `Free` and `free`. Stopword removal deletes selected common words, although some words that appear on stopword lists (especially negations) may carry important meaning. Stemming reduces related word forms to approximate roots, which can combine variants such as inflected forms and reduce vocabulary size. These choices should be tested for the task rather than applied automatically.

### 4. N-grams

An n-gram is a contiguous sequence of *n* tokens. A unigram contains one token, a bigram contains two adjacent tokens, and a trigram contains three adjacent tokens. For the sentence `not a good film`, unigrams include `not`, `good`, and `film`; bigrams include `not good` and `good film`; and a trigram can include `not a good` (depending on tokenization).

N-grams preserve limited local order that a unigram Bag-of-Words representation loses. The bigram `not good` is useful because it records the negation together with the opinion word; the unigram `good` alone does not express the same sentiment. Larger n-gram ranges can create many additional features, increasing vocabulary size, memory requirements, sparsity, and the risk of learning rare phrases that do not generalize. The useful n-gram range depends on the amount of training data and the task.

### 5. TF-IDF, its history, and scikit-learn's implementation

Term Frequency–Inverse Document Frequency (TF-IDF) is a term-weighting method. Term frequency (TF) measures how often a term occurs in a particular document. Inverse document frequency (IDF) reduces the weight of terms that occur in many documents and gives relatively more weight to terms that occur in fewer documents. The textbook form used here is:

\[
TF(t,d)=\frac{count(t,d)}{\text{number of terms in }d}
\]

\[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
\]

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

Here, `N` is the number of documents in the collection and `df(t)` is the number of documents containing term `t`. A rare term can be more useful for distinguishing one document from another than a term that appears almost everywhere. This is a statistical weighting idea, not a claim that every rare word is meaningful; rare misspellings and noise can also receive high weights.

The historical development of IDF is associated with Karen Spärck Jones's 1972 paper, “A statistical interpretation of term specificity and its application in retrieval.” Her work formalized the idea that terms appearing in relatively few documents can be more specific and useful for information retrieval. It complemented earlier term-frequency ideas, including work associated with Hans Peter Luhn. TF-IDF became an influential weighting method in information retrieval and was later used in text classification and clustering.

With default settings, scikit-learn's `TfidfVectorizer` uses raw term counts as TF, a smoothed IDF formula, and L2 normalization. Its default IDF is:

\[
IDF_{sklearn}(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

The smoothing behaves as if an extra document containing every term once had been added, avoiding zero-division issues and preventing terms found in every document from receiving an IDF of zero. The default L2 normalization divides each document vector by its Euclidean length. Therefore, scikit-learn's displayed weights generally differ from the textbook TF (document-length-normalized) and unsmoothed IDF calculation. Parameters such as `smooth_idf`, `norm`, and `sublinear_tf` can change this behavior.

TF-IDF values are weights, not probabilities. They are not constrained to sum to one as a probability distribution, and a weight such as `0.6` does not mean a 60% chance. It is a numerical feature expressing a term's relative importance under the chosen weighting and normalization rules.

### 6. `fit`, `transform`, `fit_transform`, OOV terms, and leakage

`fit` learns the representation from supplied training text. For a count vectorizer, this means learning the vocabulary and its feature indices; for TF-IDF, it also learns document-frequency information used for IDF weights. `transform` converts new documents using the already learned representation without changing its vocabulary. `fit_transform` performs both operations and is convenient for the training set.

An out-of-vocabulary (OOV) word is a token that was not present in the fitted vocabulary. A normal `CountVectorizer` or `TfidfVectorizer` ignores that token during transformation; if every token in a document is unknown or filtered out, the resulting row can be all zeros. Morphological normalization, such as stemming or lemmatization, can sometimes map related word forms to a common form, but it does not guarantee that every new word will be recognized.

Data leakage occurs when information from the held-out test set influences a preprocessing step or model fitting. If a vectorizer is fitted on both training and test documents, the test vocabulary influences the feature space before evaluation. This makes the evaluation less independent and can make results appear more reliable than they really are, even though vocabulary leakage alone does not necessarily reveal test labels. The correct sequence is: split the data first; fit the vectorizer only on training text; transform the training text and the test text with that same fitted vectorizer; then train and evaluate the model. In cross-validation, fitting should occur separately inside each training fold, commonly by using a pipeline.

### 7. African-language case study

Aryal, Prioleau, Shah, and Acharya (2023) studied sentiment classification for African languages in “Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages.” They used the AfriSenti-SemEval / NaijaSenti Twitter datasets. The paper describes 12 languages: Hausa, Yoruba, Igbo, Nigerian Pidgin, Amharic, Algerian Arabic, Moroccan Arabic/Darija, Swahili, Kinyarwanda, Twi, Mozambican Portuguese, and Xitsonga. Tweets were manually annotated as positive, negative, or neutral. The authors report 13 dataset configurations when the multilingual combination is included; they did not report Moroccan Arabic results because of technical problems with the data release.

The feature-based approach used Delta TF-IDF with conventional classifiers including SVM, LightGBM, Random Forest, and k-nearest neighbors. The paper compared these results with transformer models including AfriBERTa, AfroXLMR, and AfroLM. For example, on the Swahili test set, the Delta TF-IDF SVM weighted F1 was 0.53, while AfriBERTa and AfroXLMR each achieved 0.62. In the multilingual experiment, the Delta TF-IDF SVM test F1 was 0.63, compared with 0.69 for AfroXLMR. The authors conclude that transformer models were generally stronger on average, while Delta TF-IDF remained competitive in low-resource settings and required substantially less data.

This is relevant to the limitation of BoW and TF-IDF because these are primarily surface-form/statistical representations: they assign features to observed terms and their frequencies rather than directly representing synonymy, context, or compositional meaning. Contextual transformer models can use surrounding text to build richer representations, which helps explain why they can outperform term-frequency features. However, this study compares Delta TF-IDF with transformer models; it does not isolate synonym understanding as a separate experiment, so it should be treated as evidence about comparative task performance rather than direct proof of one specific semantic failure.

---

## Part B – Practical

**Reproducibility:** All numerical results below are generated by `vectorization_practice.py`. Run the script from this folder using `python vectorization_practice.py > output.txt`. The full console output is saved in `output.txt`.

### Task 1 – One-hot encoding

The custom encoder uses the exact vocabulary order `[cash, class, free, lunch, notes, win]`. Therefore, `free` is `[0, 0, 1, 0, 0, 0]` and `cash` is `[1, 0, 0, 0, 0, 0]`. The sentence `win free cash` produces three separate vectors because it contains three tokens; each vector still has six positions because the vocabulary contains six words.

### Task 2 – Bag of Words and document-term matrix

The fitted vocabulary contains 32 terms and the matrix shape is `(5, 32)`. Rows correspond to D1–D5, columns follow the vocabulary order below, and each cell is the count of that term in the document.

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

In D1, the `win` cell equals 2 because both `Win` and `WIN` are converted to lowercase `win` by the default vectorizer. The vectors for `win free phone` and `phone free win` are identical because BoW counts terms and ignores their order.

### Task 3 – Tokens versus vocabulary size

The vectorizer's analyzer produces 8 tokens for D1, 8 for D2, 7 for D3, 8 for D4, and 10 for D5: 41 token occurrences in total. The vocabulary size is 32. The terms appearing in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win` (each appears in two documents).

### Task 4 – Preprocessing inside the vectorizer

The default D1 vocabulary is `claim, free, now, phone, reply, to, win`. With `lowercase=False`, `FREE`, `WIN`, and `Win` remain distinct case-sensitive terms. The punctuation mark `!` is removed, and `a` is excluded because the default token pattern requires at least two word characters; English stopword removal also removes `to`. With `stop_words="english"` on D1–D5, `not` is removed, so D5 is reduced by the analyzer to `lecture, good, notes, great`; this loses the explicit negation in `not good` and can damage sentiment information.

### Task 5 – Limiting the vocabulary

The full vocabulary has 32 terms. With `min_df=2`, seven terms remain: `free, lecture, notes, now, the, today, win`; these are terms occurring in at least two documents. With `max_features=5`, the five selected terms are `free, lecture, now, the, win`, reflecting the most frequent terms under this small corpus (with ties resolved by the vectorizer's feature-selection behavior).

### Task 6 – N-grams

For D5, the unigram vocabulary has nine terms, while `(1, 2)` adds adjacent pairs such as `not good` and `were great`; these phrases retain the review's negative and positive opinion expressions. Across D1–D5, vocabulary size grows from 32 unigrams to 67 unigram/bigram features and 98 unigram/bigram/trigram features. The growth is substantial because each new n-gram can become another column, although the exact growth depends on the corpus.

### Task 7 – TF-IDF

The script prints the full TF-IDF matrix rounded to two decimal places and each non-zero D1 and D2 term with its count and TF-IDF weight. In D2 every term occurs once, but `call`, `cash`, `entry`, and `prizes` receive larger weights (0.389141) because each appears in only one document; `free`, `now`, `today`, and `win` receive 0.313957 because they occur in more than one document. In D1, `win` has the highest weight (0.573889) because it occurs twice in D1 and remains relatively distinctive across the corpus; the default TF-IDF vector is also L2-normalized.

| D2 word | Count in D2 | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.389141 |
| cash | 1 | 0.389141 |
| entry | 1 | 0.389141 |
| free | 1 | 0.313957 |
| now | 1 | 0.313957 |
| prizes | 1 | 0.389141 |
| today | 1 | 0.313957 |
| win | 1 | 0.313957 |

### Task 8 – TF-IDF by hand

D2 contains eight tokens according to the vectorizer's analyzer. Both `cash` and `today` occur once in D2, so both have TF = 1/8 = 0.125. The corpus has N = 5 documents; `cash` has df = 1 and `today` has df = 2.

| Term | Count in D2 | D2 token count | TF | df | N | IDF = ln(N/df) | Textbook TF-IDF |
|---|---:|---:|---:|---:|---:|---:|---:|
| cash | 1 | 8 | 0.125000 | 1 | 5 | ln(5/1) = 1.609438 | 0.201180 |
| today | 1 | 8 | 0.125000 | 2 | 5 | ln(5/2) = 0.916291 | 0.114536 |

The hand calculation ranks `cash` above `today`, matching scikit-learn's ranking. The numerical weights are different because the hand calculation uses the assignment's textbook formula, while scikit-learn applies smoothed IDF and default L2 normalization.

### Task 9 – Sparse matrix

The count matrix has 5 × 32 = 160 cells. It contains 39 non-zero cells and 121 zero cells, so the percentage of zeros is 121/160 × 100 = 75.62%. The matrix is stored in CSR sparse format.

### Task 10 – Fit and transform

The vectorizer is fitted only on D1–D4 and then used to transform N1 and N2. N1 has OOV terms `prize` and `your`; N2 has OOV terms `is`, `moved`, and `tomorrow` (the one-character token `to` is excluded by the default token pattern rather than treated as an OOV feature). `prize` is unknown because the training corpus contains the different surface form `prizes`; stemming or lemmatization could normalize related word forms, depending on the chosen algorithm.

### Task 11 – Data leakage

The correct vocabulary is learned from D1–D4 only. When the vectorizer is incorrectly fitted on D1–D4 plus N1 and N2, the additional test-only vocabulary terms are `is`, `moved`, `prize`, `tomorrow`, and `your`. This is leakage because the test texts have influenced the learned feature space; the test data should only be transformed using a vectorizer fitted on training data.

### Task 12 – No meaning and cosine similarity

The review vocabulary is `delicious, food, tasty, terrible, the, was`. All three vectors share the terms `the`, `food`, and `was`, while their opinion words (`tasty`, `delicious`, `terrible`) occupy separate columns. The cosine similarities are 0.75 for every distinct pair, so R1 is not more similar to R2 than to R3 according to this BoW representation; it counts shared words but does not know that “tasty” and “delicious” are semantically positive synonyms while “terrible” is negative.

### Recommended vectorizer settings for a spam detector

For this small spam-detection example, I would compare both `CountVectorizer` and `TfidfVectorizer` on a properly separated validation set rather than assume one is always better. I would start with word unigrams and add bigrams because phrases such as `free prize` can be useful spam signals, while avoiding very large n-gram ranges that would increase the vocabulary from 32 to 98 features even in this tiny corpus. I would be cautious with English stopword removal: the practical result shows that it removes `not`, which can reverse or weaken meaning, and it can remove potentially useful message terms. I would test `min_df` and `max_features` on a larger training corpus to reduce noise and memory use, but would not set them aggressively on a small dataset because rare spam indicators such as `cash` or `prizes` may be valuable. Most importantly, I would fit every vectorizer on training data only and transform validation/test messages with that fitted vectorizer.

---

## References

1. Spärck Jones, K. (1972). “A statistical interpretation of term specificity and its application in retrieval.” *Journal of Documentation*, 28(1), 11–21. https://www.cl.cam.ac.uk/archive/ksj21/ksjdigipapers/compling05.pdf
2. Aryal, S. K., Prioleau, H., Shah, U., & Acharya, S. (2023). “Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages.” *Journal of Computing Sciences in Colleges*, 39(3), 327–336. https://www.ccsc.org/publications/journals/EA2023.pdf
3. scikit-learn developers. “TfidfVectorizer.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
4. scikit-learn developers. “TfidfTransformer.” scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
5. scikit-learn developers. “Feature extraction: working with text data.” scikit-learn User Guide. https://scikit-learn.org/stable/modules/feature_extraction.html
6. Robertson, S. E., & Spärck Jones, K. (1976). “Relevance weighting of search terms.” *Journal of the American Society for Information Science*, 27, 129–146. (Historical discussion and reference details are also summarized in the British Academy profile of Karen Spärck Jones: https://www.thebritishacademy.ac.uk/documents/1705/166p273.pdf.)
