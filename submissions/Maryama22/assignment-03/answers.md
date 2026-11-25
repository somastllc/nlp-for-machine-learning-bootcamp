# Assignment 3 – Vectorization

## Part A – Research

### 1. Text Vectorization, Fixed Feature Dimensionality, and One-Hot Encoding

**Text vectorization** is the process of mapping variable-length unstructured text—such as sentences, reviews, or documents—into fixed-length numeric vectors so statistical and machine learning algorithms can compute distances, dot products, and decision boundaries over them.

Most classical machine learning estimators (such as logistic regression, linear support vector machines, and naive Bayes) operate over a geometric feature space $\mathbb{R}^d$ where every training and inference instance must align to the same coordinate axes. Because raw documents vary in word count, padding or truncating raw text without a shared index space would misalign features across examples; fixing a vocabulary of size $d$ ensures every document is represented as a point in $\mathbb{R}^d$, allowing consistent parameter weights $\mathbf{w} \in \mathbb{R}^d$ across all observations.

**One-hot encoding** represents each discrete vocabulary item $w_i$ ($i \in \{1, \dots, d\}$) as a binary basis vector $\mathbf{e}_i \in \{0, 1\}^d$ containing a single `1` at index $i$ and `0`s elsewhere. While useful as an atomic symbol representation, one-hot encoding is insufficient on its own to represent an entire document for two reasons:

1. **Variable sequence length:** A document of length $L$ produces a sequence of $L$ vectors ($\mathbb{R}^{L \times d}$), rather than a single fixed-size document vector in $\mathbb{R}^d$.
2. **Orthogonality and lack of aggregation:** Distinct one-hot vectors are mutually orthogonal ($\mathbf{e}_i^\top \mathbf{e}_j = 0$ for $i \neq j$), meaning they treat every term as equidistant and do not aggregate multi-word context into a single document-level summary without an additional pooling or counting step.

**Real-life example:** Consider a customer-support ticket system with a 4-word vocabulary `["billing", "refund", "login", "slow"]`. A short ticket `"login slow"` yields two one-hot vectors (`[0, 0, 1, 0]` and `[0, 0, 0, 1]`), whereas a longer ticket `"billing refund login slow"` yields four vectors. A standard linear classifier trained on single-token vectors cannot directly ingest variable-length lists of vectors without first aggregating them into a fixed-length representation.

---

### 2. Bag of Words and the Document-Term Matrix

The **Bag of Words (BoW)** model represents a document as an unordered multiset of tokens, recording term frequencies while discarding syntactic order and word position. Stacking the BoW vectors of $N$ documents across a vocabulary of $V$ terms forms a **document-term matrix (DTM)** $\mathbf{M} \in \mathbb{R}^{N \times V}$:

* **Rows ($i = 1, \dots, N$):** Individual documents or texts in the corpus.
* **Columns ($j = 1, \dots, V$):** Terms (features) in the learned vocabulary.
* **Cells ($M_{ij}$):** The occurrence count (or frequency) of term $j$ in document $i$.

BoW is called a *"bag"* because it acts like a container into which words are tossed without preserving sequential adjacency—just as tossing marbles into a bag forgets the order in which they were dropped.

* **Task where losing word order causes minimal harm:** **Topic categorization or broad spam filtering** based on strong topical keywords (e.g., classifying an article into sports, finance, or health using domain terms like `"inflation"`, `"equity"`, or `"portfolio"`), where term presence/frequency strongly signals subject matter.
* **Task where losing word order causes serious harm:** **Negation-sensitive sentiment analysis or relation extraction**, such as distinguishing `"the service was not good"` from `"the service was good"`, or `"bank loaned to borrower"` from `"borrower loaned to bank"`, where word order reverses polarity or actor roles.

---

### 3. Vocabulary Size, Sparsity, Pruning, and Preprocessing

In natural language processing, **tokens** are individual token instances occurring in the corpus, whereas the **vocabulary** is the set of unique types (distinct terms) retained as matrix columns; **vocabulary size** is $|V|$. **Sparsity** refers to the proportion of zero-valued entries in the document-term matrix.

Real-world corpora grow large and sparse due to **Zipf's law** and **Heaps' law**: language contains a small set of very frequent words alongside a long tail of rare words, typos, proper names, and inflected forms, while each individual document uses only a tiny fraction of the global vocabulary. Storing a dense $N \times V$ float/int array wastes memory on zeros, so libraries such as `scipy.sparse` and scikit-learn use **sparse matrix formats**—such as Compressed Sparse Row (**CSR**) or Compressed Sparse Column (**CSC**)—which store only non-zero values alongside their row/column index pointers.

Vocabulary size and feature behavior are shaped by corpus-frequency thresholds and text preprocessing:

* **`min_df` (minimum document frequency):** Drops terms appearing in fewer than a specified count or fraction of documents, filtering out ultra-rare typos and hapax legomena to reduce noise and dimensionality.
* **`max_df` (maximum document frequency):** Drops terms appearing in more than a specified count or fraction of documents, automatically filtering out corpus-wide high-frequency terms that carry little discriminative power.
* **`max_features`:** Caps the vocabulary at the top $K$ terms ranked by term frequency across the corpus (breaking ties by vocabulary order).
* **Lowercasing:** Collapses case variants (e.g., `"Policy"` and `"policy"`) into one feature, shrinking vocabulary size while occasionally losing case distinctions (e.g., `"US"` vs. `"us"`).
* **Stopword removal:** Removes high-frequency function words (such as articles and prepositions), reducing matrix width, though aggressive lists can strip words that matter in specific tasks (e.g., negations in sentiment analysis).
* **Stemming:** Heuristic suffix-stripping (e.g., Porter or Snowball stemmer) merges morphological variants sharing a root—such as `"connect"`, `"connected"`, `"connection"`, and `"connecting"`—into a single stem feature, reducing sparsity and vocabulary size at the risk of conflating unrelated words.

---

### 4. N-Grams: Capturing Local Word Order and Their Cost

An **$n$-gram** is a contiguous sequence of $n$ tokens extracted from a text: **unigrams** ($n=1$) are single tokens, **bigrams** ($n=2$) are adjacent pairs, and **trigrams** ($n=3$) are adjacent triples.

While BoW treats tokens independently, including bigrams and trigrams preserves **local phrase order** across adjacent tokens. For example, unigrams alone see `"no"`, `"refund"`, and `"available"` separately, whereas the bigram `"no refund"` captures an explicit policy condition that changes the meaning of the text compared to `"full refund"`.

The primary cost of higher-order $n$-grams is **combinatorial vocabulary explosion and increased sparsity**. A corpus of $V$ distinct unigrams can theoretically generate up to $V^2$ bigrams and $V^3$ trigrams (bounded in practice by document lengths and co-occurrence, yet still multiplying feature counts rapidly). Larger matrices require more memory, slow down training and inference, and cause many higher-order $n$-grams to appear only once or twice, increasing overfitting risk unless pruned with `min_df` or `max_features`.

---

### 5. TF-IDF: Origins, Formulas, Scikit-Learn Differences, and Weights vs. Probabilities

#### Origin of Inverse Document Frequency
The statistical rationale behind **inverse document frequency (IDF)** was introduced by **Karen Spärck Jones (1972)** in *"A statistical interpretation of term specificity and its application in retrieval"* (*Journal of Documentation*), which argued that a term's specificity—and thus its value in distinguishing relevant documents—is inversely related to how widely it is distributed across the document collection. Combined with **term frequency (TF)** (often traced to H. P. Luhn's frequency weighting ideas and later formalized alongside IDF in **TF-IDF** weighting schemes), rare terms receive higher weights because they uniquely characterize specific documents rather than appearing across the entire collection.

#### Textbook Formula and Intuition
A common textbook formulation for term $t$ in document $d$ within a collection of $N$ documents (where $\text{df}(t)$ is the number of documents containing $t$) is:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

$$\text{IDF}(t) = \log\left(\frac{N}{\text{df}(t)}\right)$$

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Rarer terms have smaller $\text{df}(t)$, making $\frac{N}{\text{df}(t)}$ and its logarithm larger; therefore, a term that appears in only a few documents receives a higher IDF multiplier than a term appearing in almost every document.

#### How Scikit-Learn's `TfidfVectorizer` Differs
Scikit-learn's `TfidfVectorizer` modifies the textbook calculation by default in two main ways:

1. **Smoothed IDF (`smooth_idf=True` by default):** It adds `1` to both numerator and denominator and adds `1` to the resulting log:
   $$\text{idf}(t) = \log\left(\frac{1 + n}{1 + \text{df}(t)}\right) + 1$$
   where $n$ is the number of training documents. Adding `1` to the denominator prevents division by zero if a term has zero document frequency in an edge case, and adding `+1` outside the logarithm ensures terms appearing in all training documents still retain a positive base weight rather than dropping to zero.
2. **Row-wise L2 normalization (`norm='l2'` by default):** After computing TF-IDF entries (where term frequency uses raw counts rather than length-normalized TF), each document row vector $\mathbf{v}$ is scaled to unit Euclidean norm ($\|\mathbf{v}\|_2 = 1$), reducing bias from raw document length differences.

#### Why TF-IDF Values Are Weights, Not Probabilities
TF-IDF scores are **heuristic importance weights**, not probabilities: they are not bounded in $[0, 1]$ prior to normalization, they do not sum to `1` across terms or classes, and a score such as `0.55` does not represent a 55% likelihood or posterior probability. Instead, a higher TF-IDF value indicates stronger relative term salience within a document given its rarity across the corpus.

---

### 6. `fit`, `transform`, `fit_transform`, Out-of-Vocabulary Terms, and Data Leakage

In scikit-learn vectorizers:

* **`fit(X_train)`:** Learns the vocabulary mapping (and IDF statistics for `TfidfVectorizer`) exclusively from the training texts `X_train`.
* **`transform(X)`:** Converts texts in `X` into a numeric matrix using the already-learned vocabulary and IDF weights, without modifying the vocabulary or adding new columns.
* **`fit_transform(X_train)`:** Performs `fit` and `transform` on `X_train` in one step so the training matrix uses the exact vocabulary learned from the training split.

When `transform` encounters **out-of-vocabulary (OOV)** words—tokens absent from the training vocabulary—those words are ignored (no new columns are created), meaning a text containing only OOV words produces an all-zero vector.

**Data leakage** occurs when information from outside the training set (such as validation or test texts) leaks into feature construction or model training. If a vectorizer is fitted on the combined dataset (`train + test`) before splitting, words that appear exclusively in the test set enter the vocabulary (and alter IDF weights and feature dimensions). This makes evaluation results look artificially optimistic because the feature space was tailored with knowledge of unseen test inputs that would not exist at deployment time.

**Correct workflow order:**
1. Split raw data into training and test sets first.
2. Call `fit_transform` on the **training** set only.
3. Call `transform` on the **test** set using the vectorizer fitted on the training set.

---

### 7. Case Study: Text Representations and the Semantic Limitation of BoW/TF-IDF

#### Study Overview
In **MasakhaNEWS** (*"MasakhaNEWS: News Topic Classification for African languages"*, Adelani et al., 2023), researchers introduce a multi-topic news classification benchmark across 16 African languages—including **Somali (`som`)**—covering categories such as *business*, *entertainment*, *health*, *politics*, *religion*, *sport*, and *technology*. The study evaluates classical bag-of-words/n-gram/TF-IDF feature representations paired with linear/statistical classifiers alongside fine-tuned pretrained language models and dense embeddings (such as XLM-R, AfriBERTa, and AfroXLM-R) across standardized train/validation/test splits.

#### How It Illustrates the Lack of Meaning in BoW and TF-IDF
BoW and TF-IDF treat every feature dimension as an independent symbol matched by exact string identity. Consequently, two texts discussing the same event or concept using different lexical choices or inflected forms—common in morphologically rich African languages such as Somali and Swahili—share little or no exact token overlap, so BoW/TF-IDF assigns zero similarity to synonyms or paraphrases unless enough co-occurring training examples bridge them. Dense pretrained embeddings and contextual encoders represent tokens and sequences in continuous vector spaces where semantically related words and cross-lingual/morphological variants lie closer together, helping capture meaning beyond exact string matching.

---

### References

1. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.
2. Robertson, S. (2004). Understanding inverse document frequency: on theoretical arguments for IDF. *Journal of Documentation*, 60(5), 503–520.
3. Adelani, D., Masiak, M., Azime, I. A., Alabi, J., et al. (2023). MasakhaNEWS: News Topic Classification for African Languages. *Proceedings of IJCNLP-AACL* / *arXiv:2304.09972*.
4. Scikit-learn Developers. *TfidfVectorizer* and *CountVectorizer* Documentation (Scikit-learn 1.9.x).

---

## Part B – Practical

### 1. Document-Term Matrix (Task 2, Shape `5 x 32`)

| Document | `are` | `before` | `but` | `call` | `cash` | `claim` | `class` | `entry` | `for` | `free` | `good` | `great` | `lecture` | `lunch` | `me` | `meeting` | `not` | `notes` | `now` | `phone` | `please` | `prizes` | `reply` | `send` | `still` | `the` | `to` | `today` | `was` | `we` | `were` | `win` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **D1** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| **D2** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **D3** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **D4** | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **D5** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

---

### 2. D2 Count vs. TF-IDF Weight Comparison (Task 7)

| Word in D2 | Count (`CountVectorizer`) | Document Frequency (`df`) | TF-IDF Weight (`TfidfVectorizer`) |
|---|---:|---:|---:|
| `call` | 1 | 1 | 0.3907 |
| `cash` | 1 | 1 | 0.3907 |
| `entry` | 1 | 1 | 0.3907 |
| `free` | 1 | 2 | 0.3153 |
| `now` | 1 | 2 | 0.3153 |
| `prizes` | 1 | 1 | 0.3907 |
| `today` | 1 | 2 | 0.3153 |
| `win` | 1 | 2 | 0.3153 |

---

### 3. Hand Calculations for D2: `cash` vs. `today` (Task 8)

Using $N = 5$ documents and $8$ extracted tokens in D2 (`free`, `entry`, `win`, `cash`, `prizes`, `today`, `call`, `now`), with $\text{TF} = \frac{\text{count in D2}}{8}$ and $\text{IDF} = \ln\left(\frac{5}{\text{df}}\right)$:

* **`cash`** ($\text{count} = 1$, $\text{df} = 1$):
  $$\text{TF}(\text{cash}, \text{D2}) = \frac{1}{8} = 0.1250$$
  $$\text{IDF}(\text{cash}) = \ln\left(\frac{5}{1}\right) = \ln(5) \approx 1.6094$$
  $$\text{TF-IDF}(\text{cash}, \text{D2}) = 0.1250 \times 1.609438 \approx \mathbf{0.2012}$$

* **`today`** ($\text{count} = 1$, $\text{df} = 2$, appearing in D2 and D3):
  $$\text{TF}(\text{today}, \text{D2}) = \frac{1}{8} = 0.1250$$
  $$\text{IDF}(\text{today}) = \ln\left(\frac{5}{2}\right) = \ln(2.5) \approx 0.9163$$
  $$\text{TF-IDF}(\text{today}, \text{D2}) = 0.1250 \times 0.916291 \approx \mathbf{0.1145}$$

---

### 4. Short Answers by Task

* **Task 1 (One-hot encoding):** Each word in `free cash` maps to a 6-element vector (`[0, 0, 1, 0, 0, 0]` and `[1, 0, 0, 0, 0, 0]`). The phrase `win free cash` produces three separate 6-dimensional vectors because each token gets its own one-hot vector, so a sentence with $k$ tokens yields $k$ vectors. A model expecting a single fixed-length vector per input cannot consume a variable-length list of word vectors directly without pooling or aggregation.

* **Task 2 (Bag of Words):** Fitting `CountVectorizer` on D1–D5 yields a `(5, 32)` document-term matrix over 32 alphabetical features. In D1, the `win` column has value `2` because `"Win"` and `"WIN"` both lowercase to `"win"` and appear twice in D1. Transforming `"win free phone"` and `"phone free win"` produces identical vectors (`True`) because BoW discards word order.

* **Task 4 (Preprocessing):** On D1 alone, default preprocessing lowercases text, drops the single-letter word `"a"` and punctuation (`!`, `.`), and merges `"Win"` and `"WIN"` into `"win"`, whereas `lowercase=False` keeps `"Win"`, `"WIN"`, and `"FREE"` separate and `stop_words="english"` removes `"to"`. On D1–D5, `stop_words="english"` removes `"not"`, which is problematic for D5 (`"The lecture was not good, but the notes were great."`) because dropping `"not"` strips explicit negation before `"good"`.

* **Task 5 (Limiting vocabulary):** The full vocabulary has 32 terms, `min_df=2` keeps 7 terms (`free`, `lecture`, `notes`, `now`, `the`, `today`, `win`), and `max_features=5` keeps 5 terms (`free`, `lecture`, `notes`, `now`, `the`). Only terms appearing in at least two documents survive `min_df=2`, while `max_features=5` retains the top-frequency terms (breaking ties in alphabetical order).

* **Task 6 (N-grams):** On D5, `ngram_range=(1, 1)` produces 9 unigrams and `(1, 2)` adds 7 bigrams (16 features total), with `"not good"` preserving the negated sentiment phrase. Across D1–D5, vocabulary size grows from 32 (`(1, 1)`) to 70 (`(1, 2)`) and 106 (`(1, 3)`), more than tripling from unigrams to up-to-trigrams on just five short sentences.

* **Task 7 (TF-IDF):** In D2, every term has count `1`, yet `call`, `cash`, `entry`, and `prizes` get higher TF-IDF weights (`0.3907`) than `free`, `now`, `today`, and `win` (`0.3153`) because the former appear in only one document while the latter appear in two. In D1, `win` gets the highest weight (`0.5961`) because it occurs twice in D1 (higher term frequency) while still appearing in only two documents overall.

* **Task 8 (TF-IDF by hand):** By hand on D2, `cash` gets `0.2012` ($\text{TF}=0.1250$, $\text{IDF}=1.6094$) and `today` gets `0.1145` ($\text{TF}=0.1250$, $\text{IDF}=0.9163$). Both the textbook formula and scikit-learn rank `cash` higher than `today` because `cash` has a lower document frequency ($\text{df}=1$ vs. $\text{df}=2$).

* **Task 10 (Fit and transform):** Fitting on D1–D4 learns 26 features; transforming N1 flags `["your", "prize"]` as out-of-vocabulary, and transforming N2 flags `["is", "moved", "tomorrow"]` as out-of-vocabulary. `"prize"` is unknown because D2 contains `"prizes"` and exact string matching treats plural and singular forms as distinct tokens; stemming or lemmatization from Assignment 2 could normalize `"prizes"` to `"prize"`.

* **Task 11 (Data leakage):** Fitting on D1–D4 yields 26 terms, whereas fitting on `D1–D4 + N1 + N2` yields 31 terms. Five test-set words—`["is", "moved", "prize", "tomorrow", "your"]`—leaked into the vocabulary when fitting included the new texts.

* **Task 12 (No meaning):** Over R1–R3 (6 vocabulary terms), every pair of reviews has a cosine similarity of `0.7500` (`[[1., 0.75, 0.75], [0.75, 1., 0.75], [0.75, 0.75, 1.]]`). R1 (`tasty`) is not more similar to R2 (`delicious`) than to R3 (`terrible`) because each review shares three neutral words (`food`, `the`, `was`) and has zero overlap on its sentiment adjective, treating `tasty`, `delicious`, and `terrible` as unrelated orthogonal features.

---

### 5. Recommended Settings for a Spam Detector

For a spam detector, I would use word and bigram features (`ngram_range=(1, 2)`), compare raw counts (`CountVectorizer`) against `TfidfVectorizer` on cross-validated training splits, keep `lowercase=True`, and apply conservative frequency pruning such as `min_df=2` (and `max_features` if needed on large corpora) so promotional phrases like `"free prize"` or repeated cues like `"win"` are captured without letting one-off typos bloat the matrix. In our outputs, `ngram_range=(1, 2)` on D5 preserved `"not good"`, while in D1 `"win"` appeared twice (`count=2`) and in D2 `"free"`, `"now"`, `"today"`, and `"win"` appeared across multiple messages; by contrast, I would avoid blind `stop_words="english"` when it strips action or urgency words (such as `"call"` or `"now"` in spam-like text or `"not"` in D5) and avoid fitting the vectorizer on test data so test vocabulary does not leak.