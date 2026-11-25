# Assignment 3 – Vectorization

---

## Part A – Research

### 1. Text Vectorization, Fixed Feature Lengths, and One-Hot Encoding

**Text vectorization** is the process of converting unstructured text (words, phrases, or documents) into fixed-length numerical vectors or matrices so that mathematical and statistical machine learning algorithms can process it. Because most classical machine learning models—such as linear regression, logistic regression, support vector machines, and k-nearest neighbors—operate on geometric and algebraic operations (dot products, matrix multiplications, and distances) in a Euclidean feature space, every input sample must be represented by the **same number of numerical features** in a **consistent order**. Without a uniform feature vector length, a trained model cannot align weights or distances across different documents.

**One-hot encoding** represents each word in a vocabulary of size $V$ as a binary vector of length $V$, containing a single `1` at the index corresponding to that word and `0` everywhere else. For example, if a small vocabulary is `["cash", "class", "free", "lunch", "notes", "win"]` ($V = 6$):
* `"free"` $\rightarrow$ `[0, 0, 1, 0, 0, 0]`
* `"cash"` $\rightarrow$ `[1, 0, 0, 0, 0, 0]`

**Why one-hot encoding is not enough for a whole document:**
1. **Variable length / sequence ambiguity:** A single one-hot vector represents *one word*, not a whole document. Representing a document as a sequence of one-hot vectors produces a matrix whose row count varies with document length, which many standard tabular/linear models do not accept directly.
2. **Loss of word order and repetition (if summed/averaged):** If you sum or average one-hot vectors across a document, you lose word order and get raw counts (which is Bag of Words).
3. **No semantic similarity (orthogonality):** Every pair of distinct one-hot vectors is orthogonal (dot product is `0`, Euclidean distance is $\sqrt{2}$). `"Tasty"` and `"delicious"` are just as far apart in one-hot space as `"tasty"` and `"terrible"`.

*Real-life example:* Suppose a customer support ticket system tracks complaints about `"refund"` and `"invoice"`. With a 4-word vocabulary `["billing", "invoice", "refund", "shipping"]`, a one-hot encoder can mark `"refund"` as `[0, 0, 1, 0]`, but a ticket saying *"I received the wrong shipping box and need a refund"* contains multiple words in a specific sequence and context that a single one-hot vector cannot capture on its own.

---

### 3. Bag of Words and the Document-Term Matrix

The **Bag of Words (BoW)** model represents a text document as an unordered multiset of its words, tracking how often each term appears while discarding grammar and word order. When stacked across a collection (corpus) of $N$ documents and a vocabulary of $V$ terms, it forms a **Document-Term Matrix (DTM)** of shape $(N, V)$:
* **Rows:** Individual documents (e.g., emails, reviews, or messages).
* **Columns:** Unique terms (features) in the corpus vocabulary.
* **Cells:** The occurrence count (or presence indicator) of a specific column term within a specific row document.

It is called a **"bag"** because it tosses all the words of a document into a bag and shakes them up—keeping track of *what* is inside and *how many* of each item are there, but forgetting the sequence in which they were pulled out.

* **Task where losing word order does little harm:** **Topic classification or spam/news categorization** (e.g., sorting news articles into *Sports*, *Finance*, or *Politics*). Words like `"goal"`, `"penalty"`, `"treasury"`, or `"casino"` strongly signal a topic regardless of exact word order.
* **Task where losing word order causes real problems:** **Sentiment analysis, negation, or relation extraction** (e.g., *"not good"* vs. *"good"*, or *"dog bites man"* vs. *"man bites dog"*). Bag of Words sees the same individual words in *"not good"* and *"good"* if negation isn't preserved, flipping or muddying polarity.

---

### 4. Vocabulary Size, Sparsity, and Vocabulary Controls

As a corpus grows, the **vocabulary size** ($V$) expands rapidly because every new document introduces new words, typos, proper nouns, and rare terms. Because any single document only uses a tiny fraction of the entire corpus vocabulary, most cells in a real-world Document-Term Matrix are `0`. This high proportion of zeros is called **sparsity**.

**Efficient storage of sparse matrices:** Instead of allocating memory for every `0` in a dense 2D array, libraries like `scipy.sparse` store only non-zero entries and their coordinates—such as **CSR (Compressed Sparse Row)** or **CSC (Compressed Sparse Column)** formats—dramatically reducing memory footprint and speeding up matrix-vector products.

**How vocabulary limits and preprocessing shape the vocabulary:**

| Setting / Choice | What It Does | Effect on Vocabulary Size & Composition |
| :--- | :--- | :--- |
| `min_df` | Drops terms appearing in fewer than `min_df` documents (or fraction of documents). | Shrinks vocabulary by removing rare typos, hapax legomena, and niche noise. |
| `max_df` | Drops terms appearing in more than `max_df` documents (or fraction). | Shrinks vocabulary by removing ultra-common corpus-wide words that carry little discriminative power. |
| `max_features` | Keeps only the top $K$ terms ranked by term frequency across the corpus. | Caps vocabulary size at $K$, retaining the most frequent terms. |
| **Lowercasing** | Converts all characters to lowercase. | Merges case variants (e.g., `"Win"` and `"win"`), reducing vocabulary size. |
| **Stopwords** | Removes high-frequency function words (e.g., `"the"`, `"is"`, `"a"`). | Shrinks vocabulary and removes common grammatical glue (though sometimes removing useful cues like `"not"`). |
| **Stemming** | Chops word suffixes heuristically to a common stem (e.g., `"prizes"` $\rightarrow$ `"prize"`). | Collapses inflectional/derivational variants into shared stems, reducing vocabulary size at some risk of over-stemming. |

---

### 5. N-Grams: Local Word Order and Trade-Offs

An **n-gram** is a contiguous sequence of $n$ tokens extracted from text:
* **Unigrams ($n = 1$):** Single words (e.g., `"not"`, `"good"`).
* **Bigrams ($n = 2$):** Adjacent word pairs (e.g., `"not good"`).
* **Trigrams ($n = 3$):** Adjacent word triples (e.g., `"was not good"`).

Using `ngram_range=(1, 2)` or `(1, 3)` preserves **local word order** across short spans, allowing models to capture phrases and modifiers rather than isolated words alone.

*Example where a bigram changes meaning:* In the review *"The lecture was not good, but the notes were great"*, unigrams alone see `"good"` and `"not"` separately. The bigram `"not good"` explicitly captures negation, changing the meaning from positive praise to a negative evaluation of the lecture.

*Cost of larger n-grams:* Increasing $n$ causes **combinatorial vocabulary explosion**, higher sparsity, longer training/inference times, and greater risk of overfitting on phrases that appear only once or twice in the training set.

---

### 6. TF-IDF: Origins, Formula, Scikit-Learn Differences, and Weights vs. Probabilities

**TF-IDF** stands for **Term Frequency–Inverse Document Frequency**, a numerical weighting scheme that boosts terms frequent in a *specific* document while penalizing terms common across *many* documents in the corpus.

**Origins of IDF:** Inverse Document Frequency was introduced by **Karen Spärck Jones (1972)** in *"A statistical interpretation of term specificity and its application in retrieval"* (*Journal of Documentation*), who proposed weighting index terms inversely to the number of documents in which they occur—formalizing the intuition that rare terms have higher specificity for distinguishing relevant documents.

**Textbook formula:** A common textbook form defines term frequency $\text{tf}(t, d)$ (raw count or normalized frequency of term $t$ in document $d$) and inverse document frequency for a corpus of $N$ documents where $\text{df}(t)$ documents contain term $t$:

$$\text{idf}(t) = \ln\left(\frac{N}{\text{df}(t)}\right), \qquad \text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

Because $\text{df}(t) \le N$, rarer terms have a smaller denominator $\text{df}(t)$, producing a larger ratio $\frac{N}{\text{df}(t)}$ and thus a **higher IDF weight**.

**How scikit-learn's `TfidfVectorizer` differs from the textbook formula:**
1. **Smoothed IDF (`smooth_idf=True` by default):** Scikit-learn computes:
   $$\text{idf}(t) = \ln\left(\frac{1 + n}{1 + \text{df}(t)}\right) + 1$$
   (where $n$ is the number of documents), adding `1` to numerator and denominator (preventing division by zero and avoiding a zero IDF if a term appeared in every document) and adding `+ 1` so even terms present in all documents retain a positive weight.
2. **Row normalization (`norm='l2'` by default):** After multiplying term counts by IDF, scikit-learn normalizes each document's TF-IDF vector so its Euclidean ($L_2$) norm equals `1` (making dot products between document vectors equal to cosine similarity).
3. **Optional sublinear TF (`sublinear_tf=False` by default):** When set to `True`, it replaces raw count $\text{tf}$ with $1 + \ln(\text{tf})$.

**Why TF-IDF values are weights, not probabilities:** A probability distribution over features or terms must be non-negative and sum to `1` across mutually exclusive outcomes. TF-IDF scores are unbounded heuristic relevance weights (prior to L2 normalization, and even after L2 normalization they are unit-norm coordinates, not probabilities). A higher TF-IDF score means higher relative importance/specificity in that document under the weighting formula, not the probability that a word occurs or that a document belongs to a class.

---

### 7. `fit`, `transform`, `fit_transform`, Out-of-Vocabulary Words, and Data Leakage

* **`fit(X_train)`:** Learns the vocabulary (and document frequencies/IDF weights for TF-IDF) exclusively from the training texts `X_train`.
* **`transform(X)`:** Maps texts in `X` into numerical vectors using the *already learned* vocabulary and weights from `fit`.
* **`fit_transform(X_train)`:** Performs `fit` on `X_train` and immediately returns `transform(X_train)` in one step.

**Out-of-vocabulary (OOV) words:** When `transform` encounters a word not seen during `fit`, a standard fitted `CountVectorizer` or `TfidfVectorizer` simply **ignores/drops** that token—it does not add new columns after fitting.

**Data leakage in vectorization:** Data leakage occurs when information from outside the training set (such as test or validation texts) influences the feature representation—for example, fitting a vectorizer on `train + test` before splitting or evaluating. Fitting on test text leaks test vocabulary existence and corpus document frequencies into the feature space, making evaluation metrics look artificially optimistic because the feature representation already "saw" the test distribution.

**Correct order of steps:**
1. Split data into **train** and **test** sets first.
2. Call **`fit_transform`** (or `fit` then `transform`) **only on the training set**.
3. Call **`transform`** (never `fit` or `fit_transform`) **on the test set** (and on new unseen production data).

---

### 8. Case Study: Text Representations in African-Language NLP

A relevant study on text representation in an African language is **"Resource-efficient somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus"** (*Scientific Reports*, September 2026), alongside broader African-language NLP work such as **MasakhaNEWS** (Adelani et al., 2023) and morphological/neural studies on agglutinative African languages like Kiswahili (Wandwi & Mtesigwa, 2025).

* **Task:** Supervised news topic/category classification on Somali news articles (comparing classical sparse TF-IDF + Linear SVM against neural and transformer-based representations).
* **Data:** A curated corpus of Somali-language news articles.
* **Representations compared:** Sparse **TF-IDF** lexical features (paired with a Linear Support Vector Machine) evaluated alongside neural/transformer-based representations across news categories.
* **Results & Limitation of BoW/TF-IDF:** TF-IDF + Linear SVM provides a strong, transparent, and computationally light baseline for Somali news classification when resources are limited. However, because Bag of Words and TF-IDF treat tokens as discrete index dimensions based on surface string matches and corpus counts, they **do not capture semantic meaning, synonymy, or context**. For instance, in an agglutinative language like Somali—where words take rich morphological forms and can vary in spelling or phrasing (such as *ra'iisulwasaare* vs. *raysulwasaare* in Somali IR work)—exact term matching misses semantic relatedness between distinct surface forms that dense contextual/neural embeddings can better relate.

---

## Part B – Practical Report

### Document-Term Matrix (D1–D5, `CountVectorizer` Default)

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| **D2** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **D3** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **D4** | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **D5** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

---

### D2 Word Counts vs. Scikit-Learn TF-IDF Weights

For `D2` (`Free entry: win cash prizes today. Call now.`), every present term occurs once (`count = 1`), while scikit-learn's `TfidfVectorizer` (default `smooth_idf=True`, `norm='l2'`, $n=5$) assigns higher weights to terms with `df = 1` (`0.39` rounded) than to terms with `df = 2` (`0.31` rounded):

| Word in D2 | Count (`tf`) | Document Frequency (`df`) | Scikit-Learn TF-IDF (Rounded to 2 Decimals) |
| :--- | :---: | :---: | :---: |
| `call` | 1 | 1 | 0.39 |
| `cash` | 1 | 1 | 0.39 |
| `entry` | 1 | 1 | 0.39 |
| `free` | 1 | 2 | 0.31 |
| `now` | 1 | 2 | 0.31 |
| `prizes` | 1 | 1 | 0.39 |
| `today` | 1 | 2 | 0.31 |
| `win` | 1 | 2 | 0.31 |

---

### Hand Calculations for D2 (`cash` and `today`)

Using $N = 5$ documents in D1–D5 and natural logarithm $\text{IDF} = \ln(N / \text{df})$:
* **`cash`:** $\text{tf} = 1$, $\text{df} = 1$ (only in D2). $\text{IDF} = \ln(5 / 1) = \ln(5) \approx 1.6094$. $\text{TF-IDF} = 1 \times 1.6094 = 1.6094$.
* **`today`:** $\text{tf} = 1$, $\text{df} = 2$ (in D2 and D3). $\text{IDF} = \ln(5 / 2) = \ln(2.5) \approx 0.9163$. $\text{TF-IDF} = 1 \times 0.9163 = 0.9163$.
* **Comparison with scikit-learn:** Both methods rank `cash` higher than `today` (`1.6094 > 0.9163` by hand; `0.39 > 0.31` in scikit-learn), because `cash` appears in only 1 document whereas `today` appears in 2.

---

### Short Answers by Task

* **Task 1 (One-hot encoding):** Each word in the 6-word vocabulary maps to a length-6 binary vector (`free` $\rightarrow$ `[0, 0, 1, 0, 0, 0]`, `cash` $\rightarrow$ `[1, 0, 0, 0, 0, 0]`). Encoding `win free cash` produces three separate one-hot vectors, one per word in the phrase. The number of vectors equals the sentence/token length because each token position gets its own indicator vector.
* **Task 2 (Bag of Words):** Fitting `CountVectorizer` on D1–D5 gives a 32-term vocabulary and a `(5, 32)` matrix. In D1, the cell for `win` has value `2` because `"Win"` and `"WIN"` both appear and are lowercased to `"win"`. Transforming `"win free phone"` and `"phone free win"` produces identical count vectors (`1`s at `free`, `phone`, and `win`, and `0` elsewhere) because Bag of Words ignores word order.
* **Task 4 (Tokens vs. vocabulary size):** Across D1–D5, there are `41` total tokens (`8 + 8 + 7 + 8 + 10`) and a vocabulary size of `32` unique terms. The 7 words appearing in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win`.
* **Task 5 (Preprocessing inside vectorizer):** On D1 alone, default lowercases and drops single-letter `"a"` and punctuation `"!"`, collapsing `"Win"` and `"WIN"` into `"win"`; `lowercase=False` keeps `"win"` and `"WIN"` as separate features; `stop_words="english"` removes `"a"` and `"to"`. On D1–D5 with `stop_words="english"`, `"not"` is **not** removed, which is helpful for D5 (`"was not good"`) because negation remains visible rather than vanishing.
* **Task 6 (Limiting the vocabulary):** Out of `32` total terms, `min_df=2` keeps `7` words (`free`, `lecture`, `notes`, `now`, `the`, `today`, `win`)—specifically terms occurring in at least two documents. `max_features=5` keeps `5` words—the top 5 most frequent terms across the corpus (`the`, `win`, `free`, `lecture`, `notes`).
* **Task 7 (N-grams):** On D5, `ngram_range=(1, 1)` yields unigrams while `(1, 2)` adds bigrams including `"not good"`, which preserves the negative opinion toward the lecture. Across D1–D5, vocabulary size grows from `32` for `(1, 1)` to `75` for `(1, 2)` and `117` for `(1, 3)`—roughly adding around 42–43 new n-gram features per step even on a tiny 5-document corpus.
* **Task 8 (TF-IDF):** In D2, every word has count `1`, yet `call`, `cash`, `entry`, and `prizes` get higher TF-IDF weights (`0.39`) than `free`, `now`, `today`, and `win` (`0.31`) because `df = 1` gives a higher IDF than `df = 2`. In D1, `win` gets the highest weight (`0.57`) because it occurs twice (`tf = 2`), outweighing single-occurrence `df = 1` words (`0.36`).
* **Task 10 (Sparse matrix):** The `(5, 32)` count matrix has `160` total cells (`5 * 32`), `41` non-zero cells (matching the 41 token occurrences), and `119` zero cells. That equals **74.38% zeros** (`119 / 160`).
* **Task 11 (Fit and transform):** Fitting on D1–D4 and transforming N1 (`"Claim your free prize now"`) and N2 (`"Is the lunch meeting moved to tomorrow?"`) flags OOV words `your` and `prize` in N1, and `is`, `moved`, and `tomorrow` in N2. `prize` is unknown because D2 contains plural `"prizes"`, not singular `"prize"`; stemming or lemmatization from text preprocessing could normalize `"prizes"` and `"prize"` to a shared root/lemma.
* **Task 12 (Data leakage):** Fitting on D1–D4 alone yields 25 terms, whereas fitting on `D1–D4 + N1 + N2` yields 30 terms. The 5 words that leak from the test set into the vocabulary are `is`, `moved`, `prize`, `tomorrow`, and `your`.
* **Task 13 (No meaning):** On R1–R3, the 6-word vocabulary is `['delicious', 'food', 'tasty', 'terrible', 'the', 'was']`, and every pairwise cosine similarity equals `0.75`. R1 (`tasty`) is *not* more similar to R2 (`delicious`) than to R3 (`terrible`) under count vectors because each review shares the exact same three context words (`the`, `food`, `was`) and has one orthogonal sentiment adjective (`0` dot-product overlap).

---

### Spam Detector Recommendation

For a spam detector, I would use **`TfidfVectorizer` (or `CountVectorizer` with `ngram_range=(1, 2)`)**, keep **`min_df`** modestly low on larger datasets so rare promotional/scam terms aren't discarded too aggressively, and **avoid** heavy aggressive stopword stripping that drops function words or negation/phrasing cues—while strictly **avoiding** fitting the vectorizer on test/unseen messages so test vocabulary doesn't leak. In our D1–D5 outputs, spam-like terms (`win`, `free`, `claim`, `cash`, `prizes`, `call`, `reply`) cleanly separate from academic/work messages (`lecture`, `notes`, `class`, `lunch`, `meeting`), and keeping bigrams (`ngram_range=(1, 2)`, which expanded vocabulary from 32 to 75 on D1–D5) helps preserve multi-word phrases and negations rather than treating words as isolated tokens.

---

### References

1. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Jurafsky, D., & Martin, J. H. (2024/2025). *Speech and Language Processing* (3rd ed.).
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. Adelani, D. I., et al. (2023). MasakhaNEWS: News Topic Classification for African Languages. *Proceedings of EACL*.
6. Resource-efficient somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus. *Scientific Reports* (2026).