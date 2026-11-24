# Assignment 3 – Text Vectorization

## Part A – Research

### 1. Text vectorization and one-hot encoding

**Text vectorization** is the process of converting text into a numerical representation so that a machine-learning algorithm can process it. A computer model generally operates on numerical feature arrays rather than raw sentences. For a conventional tabular model, each example must have the same number of features because the model learns a fixed set of feature positions and parameters. If one document had 20 features and another had 25, the model would not know how to align the corresponding positions.

A **one-hot vector** represents one word using a vector whose length equals the vocabulary size. The position assigned to that word is `1` and all other positions are `0`. For example, if the vocabulary is `['cash', 'class', 'free', 'lunch', 'notes', 'win']`, then `free` is `[0, 0, 1, 0, 0, 0]`. The same vocabulary and feature order must be used for every word.

One-hot encoding a word is not, by itself, a complete fixed-length representation of a document. A sentence such as `"free cash"` produces two separate word vectors, whereas `"win free cash"` produces three word vectors. Because the number of vectors changes with sentence length, a document-level model needs a consistent fixed-length representation—such as Bag of Words (counting vocabulary terms)—rather than a variable list of independent word vectors.

### 2. Bag of Words and the document-term matrix

**Bag of Words (BoW)** represents a document by recording which vocabulary terms occur and how many times each term occurs, without preserving the original word order. A **document-term matrix (DTM)** is the collection of these vectors arranged in a table:

- Each **row** represents one document.
- Each **column** represents one vocabulary term (feature).
- Each **cell** contains the count of that term in that document (or `0` when the term does not occur).

It is called a "bag" because words are treated like items placed in a bag: their counts matter, while their sequence is discarded. This works well for broad topic classification, but it can distort sentiment or intent—for instance, `"not good"` loses its negation when split into separate unordered counts, and `"win free phone"` produces the exact same vector as `"phone free win"`.

### 3. Vocabulary size and sparsity

The **vocabulary** is the set of distinct tokens retained as features after tokenization and preprocessing. **Vocabulary size** is the number of those distinct features, which differs from the total number of token occurrences across the corpus.

Because an individual document uses only a small fraction of the full corpus vocabulary, most entries in a DTM are `0`. This property is called **sparsity**. Sparse matrix formats such as Compressed Sparse Row (CSR) store only non-zero values and their positions, significantly reducing memory usage and speeding up computation.

| Setting or operation | Effect on the vocabulary |
| --- | --- |
| `min_df` | Removes terms appearing in fewer than the specified number or proportion of documents. |
| `max_df` | Removes terms appearing in more than the specified number or proportion of documents. |
| `max_features` | Keeps only the highest-frequency terms up to a specified feature count. |
| Lowercasing | Combines case variants such as `Win` and `WIN` into one feature (`win`). Disabling it keeps them separate. |
| Stopword removal | Removes selected common words. This reduces noise, though generic lists may remove meaningful words such as `not`. |
| Token length (`>= 2` chars) | By default, `CountVectorizer` ignores single-character tokens (such as `a`). |

### 4. N-grams

An **n-gram** is a consecutive sequence of *n* tokens. A unigram contains one token (`not`), a bigram contains two consecutive tokens (`not good`), and a trigram contains three. Unlike unigram BoW, n-grams preserve local word order. For example, the bigram `not good` retains the negated sentiment of `"The lecture was not good"`. However, expanding the n-gram range substantially increases the number of features, memory footprint, and matrix sparsity.

### 5. TF-IDF and the history of inverse document frequency

**Term frequency–inverse document frequency (TF-IDF)** combines how often a term appears in a single document with how rare that term is across the entire collection. The concept of inverse document frequency (IDF) was introduced by Karen Spärck Jones (1972) based on the insight that terms occurring in few documents are more discriminative than terms appearing everywhere.

The standard textbook formulation is:

$$TF(t,d) = \frac{f_{t,d}}{\text{number of terms in } d}$$

$$IDF(t) = \ln\left(\frac{N}{df(t)}\right)$$

$$TFIDF(t,d) = TF(t,d) \times IDF(t)$$

where $f_{t,d}$ is the count of term $t$ in document $d$, $N$ is the total number of documents, and $df(t)$ is the document frequency of $t$.

By default, scikit-learn's `TfidfVectorizer` uses a smoothed IDF:

$$IDF_{\text{sklearn}}(t) = \ln\left(\frac{1+N}{1+df(t)}\right) + 1$$

and applies L2 normalization to each document vector so that every row has unit Euclidean norm. Consequently, scikit-learn's exact values differ from the textbook hand calculation, though the relative ranking of terms within a document remains consistent. Additionally, TF-IDF values are **weights, not probabilities**—they measure relative term importance rather than probabilities summing to 1.

### 6. `fit`, `transform`, `fit_transform`, OOV words, and data leakage

- `fit()` learns the vocabulary and feature mapping from the training documents.
- `transform()` maps new documents into vectors using the already fitted vocabulary.
- `fit_transform()` performs both operations together on the training set.

An **out-of-vocabulary (OOV)** word is a token not present in the fitted vocabulary; `CountVectorizer` ignores OOV words during `transform()`. **Data leakage** occurs when validation or test data influences preprocessing steps (such as fitting the vocabulary on combined train and test data), producing overly optimistic evaluation results. To prevent leakage, always split the data first, fit the vectorizer strictly on training documents, and call `transform()` on validation and test documents.

### 7. Case study: text representations for African-language sentiment classification

Aryal, Prioleau, Shah, and Acharya (2023) compared **Delta TF-IDF** with conventional classifiers (SVM, LightGBM, Random Forest, KNN) against African-language **transformer** models (AfriBERTa, AfroXLMR, AfroLM) on AfriSenti / NaijaSenti datasets across 12 African languages. While transformers achieved higher average performance (multilingual test weighted F1 scores of `0.69` for AfroXLMR, `0.68` for AfriBERTa, and `0.64` for AfroLM versus `0.63` for the best feature-based SVM), Delta TF-IDF remained competitive in low-resource scenarios with limited training data and compute. Nevertheless, frequency-based methods rely on token statistics rather than deep contextual semantics, so they cannot inherently recognize synonyms (`tasty` vs. `delicious`) or compositional negation (`not good`) without explicit n-grams.

---

## Part B – Practical

### Reproducibility

All outputs below match the execution of `vectorization_practice.py` using NumPy and scikit-learn:

```bash
pip install scikit-learn numpy
python vectorization_practice.py > output.txt
```

### Task 1 – One-hot encoding

Using the vocabulary `['cash', 'class', 'free', 'lunch', 'notes', 'win']`:

| Sentence | Word | One-hot vector |
| --- | --- | --- |
| `"free cash"` | `free` | `[0, 0, 1, 0, 0, 0]` |
| `"free cash"` | `cash` | `[1, 0, 0, 0, 0, 0]` |
| `"win free cash"` | `win` | `[0, 0, 0, 0, 0, 1]` |
| `"win free cash"` | `free` | `[0, 0, 1, 0, 0, 0]` |
| `"win free cash"` | `cash` | `[1, 0, 0, 0, 0, 0]` |

The sentence `"win free cash"` produces **3** separate word vectors. Because the number of vectors varies with sentence length, one-hot word vectors alone do not produce a fixed-size document vector.

### Task 2 – Bag of Words / Document-term matrix

Fitting `CountVectorizer()` on D1–D5 produces a matrix of shape **(5, 32)** with the following 32 vocabulary features:

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

- **D1 cell with value 2:** `win -> 2`, because `"Win"` and `"WIN"` are both lowercased to `win` and summed.
- **Word order invariance:** `'win free phone'` and `'phone free win'` produce identical vectors because unigram BoW ignores word order.

### Task 3 – Tokens vs. vocabulary

- **Total tokens (length $\ge 2$):** `41`
- **Vocabulary size:** `32`
- **Words appearing in more than one document (each in 2 documents):** `free`, `lecture`, `notes`, `now`, `the`, `today`, `win`.

### Task 4 – Preprocessing

Testing on `message = ["FREE Prize! Call the office NOW"]`:

| Configuration | Extracted features |
| --- | --- |
| Default | `['call', 'free', 'now', 'office', 'prize', 'the']` |
| `lowercase=False` | `['Call', 'FREE', 'NOW', 'Prize', 'office', 'the']` |
| `stop_words='english'` | `['free', 'office', 'prize']` |

When `CountVectorizer(stop_words="english")` is fitted on D1–D5, the retained vocabulary is:
`['cash', 'claim', 'class', 'entry', 'free', 'good', 'great', 'lecture', 'lunch', 'meeting', 'notes', 'phone', 'prizes', 'reply', 'send', 'today', 'win']`.

- **Is `'not'` in vocabulary?** `False`. Removing `'not'` is risky for sentiment analysis because D5 (`"The lecture was not good, but the notes were great."`) loses its negation, leaving only `'good'`.

### Task 5 – Limiting the vocabulary

| Configuration | Vocabulary | Size |
| --- | --- | --- |
| Full vocabulary | All 32 terms | `32` |
| `min_df=2` | `['free', 'lecture', 'notes', 'now', 'the', 'today', 'win']` | `7` |
| `max_features=5` | `['free', 'lecture', 'now', 'the', 'win']` | `5` |
| `max_df=0.4` | All 32 terms (since maximum document frequency in D1–D5 is $2/5 = 0.4$) | `32` |

### Task 6 – N-grams

On D5 (`"The lecture was not good, but the notes were great."`):
- **Unigrams `(1, 1)`:** `['but', 'good', 'great', 'lecture', 'not', 'notes', 'the', 'was', 'were']`
- **Unigrams + Bigrams `(1, 2)`:** Includes pairs such as `'not good'`, `'was not'`, `'were great'`, `'lecture was'`, `'notes were'`, etc. The bigram `'not good'` preserves the negative sentiment.

Vocabulary growth across D1–D5:

| N-gram range | Number of features |
| --- | --- |
| `(1, 1)` | `32` |
| `(1, 2)` | `67` |
| `(1, 3)` | `98` |

### Task 7 – TF-IDF

Comparison of raw counts and `TfidfVectorizer` weights for D2:

| Word in D2 | Count | TF-IDF weight |
| --- | --- | --- |
| `call` | 1 | `0.3891` |
| `cash` | 1 | `0.3891` |
| `entry` | 1 | `0.3891` |
| `free` | 1 | `0.3140` |
| `now` | 1 | `0.3140` |
| `prizes` | 1 | `0.3891` |
| `today` | 1 | `0.3140` |
| `win` | 1 | `0.3140` |

Although every word in D2 has a count of `1`, `call`, `cash`, `entry`, and `prizes` occur in only 1 document (`df = 1`), giving them higher IDF weights (`0.3891`) than `free`, `now`, `today`, and `win` (`df = 2`, weight `0.3140`).

### Task 8 – TF-IDF by hand

For D2 with $N = 5$ documents and $8$ words in D2 (`free`, `entry`, `win`, `cash`, `prizes`, `today`, `call`, `now`):

| Word | Count | $df$ | $TF = \text{count} / 8$ | $IDF = \ln(5 / df)$ | $TF\text{-}IDF$ |
| --- | --- | --- | --- | --- | --- |
| `cash` | 1 | 1 | `0.125` | `1.609438` | `0.20118` |
| `today` | 1 | 2 | `0.125` | `0.916291` | `0.114536` |

Both the hand calculation and scikit-learn rank `cash` higher than `today`, though the exact numbers differ due to scikit-learn's smoothed IDF and L2 normalization.

### Task 9 – Sparse matrix

- **Shape:** `(5, 32)`
- **Total cells:** `160`
- **Non-zero cells:** `39`
- **Zero cells:** `121`
- **Percentage of zeros:** `75.62 %`

### Task 10 – Fit on training / transform new text

Fitting `CountVectorizer()` on `train_documents = documents[:4]` produces a training vocabulary of **26** features:

- `"Claim your free prize now"` -> OOV words: `['your', 'prize']` (`prize` is OOV because `prizes` is plural and no stemming/lemmatization was applied).
- `"Is the lunch meeting moved to tomorrow?"` -> OOV words: `['is', 'moved', 'tomorrow']`.

### Task 11 – Data leakage

Fitting on `train_documents + new_texts` expands the vocabulary from 26 to 31 words, leaking 5 test-only words into the training vocabulary: **`is`**, **`moved`**, **`prize`**, **`tomorrow`**, and **`your`**. To prevent leakage, always call `fit()` only on the training set and `transform()` on new/test text.

### Task 12 – Bag of Words does not capture meaning

For `reviews`:
- Vocabulary: `['delicious', 'food', 'tasty', 'terrible', 'the', 'was']`
- Cosine similarity matrix:
  $$\begin{bmatrix} 1.00 & 0.75 & 0.75 \\ 0.75 & 1.00 & 0.75 \\ 0.75 & 0.75 & 1.00 \end{bmatrix}$$
- `R1 vs R2`: `0.75` | `R1 vs R3`: `0.75`
- **Is R1 more similar to R2 than R3?** `False`. Bag of Words treats `tasty`, `delicious`, and `terrible` as completely orthogonal dimensions and does not capture semantic similarity.

### Recommended vectorizer settings for a spam detector

1. **Weighting:** Start with `TfidfVectorizer` (and compare against `CountVectorizer` on a validation set) to highlight discriminative spam terms like `cash`, `claim`, and `prizes` while capturing repeated spam triggers like `win`.
2. **N-grams:** Use `ngram_range=(1, 2)` to capture phrases such as `free phone`, `cash prizes`, and `not good`, while avoiding trigrams initially to prevent feature explosion (`32` -> `98` features).
3. **Stopwords & Vocabulary Limits:** Avoid blindly removing all English stopwords so negations like `not` are preserved, and tune `min_df` / `max_df` carefully on validation data rather than discarding rare spam indicators.
4. **Leakage Prevention:** Fit the vectorizer exclusively on the training split and apply `.transform()` to validation and test splits.

---

## References

1. Jones, K. S. (1972). "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." *Journal of Documentation*, 28(1), 11–21. https://doi.org/10.1108/eb026526
2. Salton, G., Wong, A., & Yang, C. S. (1975). "A Vector Space Model for Automatic Indexing." *Communications of the ACM*, 18(11), 613–620. https://doi.org/10.1145/361219.361220
3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
4. scikit-learn developers. "Feature extraction: Text feature extraction — TF-IDF term weighting." https://scikit-learn.org/stable/modules/feature_extraction.html
5. Aryal, S. K., Prioleau, H., Shah, U., & Acharya, S. (2023). "Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages." *Journal of Computing Sciences in Colleges*, 39(3), 327–336. https://ccsc-eastern.github.io/files/ccsce23/EA2023.pdf
6. Martineau, J., & Finin, T. (2009). "Delta TFIDF: An Improved Feature Space for Sentiment Analysis." *Proceedings of the International AAAI Conference on Web and Social Media*, 3(1), 258–261. https://ojs.aaai.org/index.php/ICWSM/article/view/139