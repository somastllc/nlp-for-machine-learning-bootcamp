## Part A – Research

### 1. Text Vectorization and One-Hot Encoding

**Text vectorization** is the process of turning raw text into numbers that a machine learning algorithm can do math on. Most standard models—such as logistic regression, support vector machines, or linear classifiers—expect every input example to be a vector of the **same fixed length**. A message or review can have three words or fifty words, so we cannot feed raw variable-length strings straight into a fixed feature table. Using a shared vocabulary of fixed positions lets every document map to a vector of the exact same size (Jurafsky & Martin, 2024).

**One-hot encoding** of single words gives each word in a vocabulary its own list of zeros with a single `1` at that word's index. Suppose our vocabulary is `["free", "prize", "class", "notes"]`. Then `"free"` becomes `[1, 0, 0, 0]` and `"prize"` becomes `[0, 1, 0, 0]`.

The catch is that one-hot encoding represents **one word at a time**, not a full document. If a customer writes `"free prize today"` (3 words), you get a sequence of 3 vectors; if another writes `"please send class notes before the exam"` (7 words), you get 7 vectors. Because classical models need a single fixed-length feature vector per document—and because one-hot vectors for different words are orthogonal (no shared overlap)—one-hot encoding alone does not summarize a whole document into one uniform input (Manning et al., 2008).

---

### 2. Bag of Words and the Document-Term Matrix

**Bag of Words (BoW)** turns a whole document into a single count vector over a chosen vocabulary. When you stack the vectors for all documents together, you get a **document-term matrix**:

* **Rows** represent individual documents (emails, tweets, reviews, or messages).
* **Columns** represent words (or terms) in the vocabulary.
* **Cells** hold how many times a given word appears in a given document (or a weight, depending on the representation).

It is called a **"bag"** because it throws away word order and grammar—like tossing words into a bag and counting what is inside.

* **Where losing word order causes little harm:** Keyword-heavy **spam or topic filtering**. If an email contains `"free"`, `"cash"`, and `"prize"` in almost any order, those words already point strongly toward spam.
* **Where losing word order causes real problems:** **Sentiment analysis or negation**, such as `"this phone is not good"` versus `"this phone is good, not bad"`. A plain bag of words sees `"not"` and `"good"` as separate counts and misses how negation attaches to specific words (Jurafsky & Martin, 2024).

---

### 3. Vocabulary Size, Sparsity, and Vocabulary Controls

**Vocabulary size** is simply the number of unique terms in the vocabulary (which equals the number of columns in the document-term matrix). **Sparsity** means that most cells in the matrix are `0`.

In real datasets, matrices get huge and mostly zero for two reasons: first, natural language has a long tail of rare words, typos, names, and numbers, so adding documents keeps introducing new words (Manning et al., 2008); second, each individual message or short post only uses a tiny fraction of all words in the dataset. Storing millions of zeros in a dense grid wastes memory, so libraries like `scipy` and `scikit-learn` use **sparse matrix formats** (such as CSR, Compressed Sparse Row) that store only the row/column coordinates and values of non-zero entries.

You can control vocabulary growth through vectorizer settings and preprocessing:

* **`min_df`:** Drops words that appear in fewer than a set number of documents (or a fraction of documents), clearing out rare typos and one-off terms.
* **`max_df`:** Drops words that appear in too many documents (above a count or fraction), filtering out overly common corpus-wide words.
* **`max_features`:** Caps the vocabulary at the top $K$ most frequent terms.
* **Preprocessing (`lowercase`, stopwords, stemming):** Lowercasing merges `"Win"` and `"win"` into one column; removing stopwords cuts common function words; stemming or lemmatization groups word variants (like `"prizes"` and `"prize"`) into shared stems or lemmas, shrinking the column count further.

---

### 4. N-Grams

An **n-gram** is a contiguous sequence of $n$ tokens from text:
* **Unigrams ($n=1$):** Single words (e.g., `"not"`, `"good"`).
* **Bigrams ($n=2$):** Pairs of neighboring words (e.g., `"not good"`).
* **Trigrams ($n=3$):** Triples of neighboring words (e.g., `"not very good"`).

By keeping adjacent tokens glued together, bigrams and trigrams preserve local word order that a unigram bag of words drops. For example, `"not worth buying"` as unigrams has `"not"`, `"worth"`, and `"buying"`, whereas the bigram `"not worth"` keeps the negation attached to `"worth"`.

The trade-off is **feature explosion**. Every distinct pair or triple of neighboring words becomes a new column, so moving from `(1, 1)` to `(1, 2)` or `(1, 3)` multiplies vocabulary size, increases sparsity, and can make models overfit rare phrases if the dataset is small.

---

### 5. TF-IDF: Origins, Formula, Scikit-Learn Differences, and Weights vs. Probabilities

**TF-IDF** stands for **Term Frequency–Inverse Document Frequency**. The idea of **inverse document frequency (IDF)** was introduced by **Karen Spärck Jones (1972)** in information retrieval to measure "term specificity"—the intuition that a word appearing in almost every document is a poor way to tell documents apart, while a word appearing in only a few documents carries more distinguishing weight.

In the textbook formulation for term $t$ in document $d$ across a collection of $N$ documents where $\text{df}(t)$ is the number of documents containing $t$:

$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{total terms in } d}$$

$$\text{IDF}(t) = \log\!\left(\frac{N}{\text{df}(t)}\right)$$

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Rarer words get higher weights because when $\text{df}(t)$ is small, $N / \text{df}(t)$ is large, so $\log(N / \text{df}(t))$ is larger than for common words found in many documents.

**How `scikit-learn`'s `TfidfVectorizer` differs:** By default (`smooth_idf=True`, `norm='l2'`, `sublinear_tf=False`), scikit-learn uses raw term counts rather than dividing by document length, smooths IDF by adding `1` to numerator and denominator and `1` to the log ($\log((1 + N) / (1 + \text{df})) + 1$), and normalizes each document's output row so its Euclidean (L2) norm equals `1`. That changes the raw numbers compared to the basic textbook formula, though rarer terms still rank higher than common ones within the same document.

**Why TF-IDF values are weights, not probabilities:** TF-IDF scores do not come from a probability distribution over outcomes and do not sum to `1` across classes or words (especially after L2 normalization). A weight of `0.55` means a term has a relatively strong weighted presence in that document—not a `55%` probability that the document is spam or belongs to a class.

---

### 6. `fit`, `transform`, `fit_transform`, Out-of-Vocabulary Words, and Data Leakage

* **`fit`:** Learns the vocabulary (and IDF statistics, for TF-IDF) from the provided texts without outputting a transformed matrix.
* **`transform`:** Converts texts into feature vectors using the vocabulary already learned during `fit`. It never adds new columns.
* **`fit_transform`:** Runs `fit` and `transform` in one step on the **training** data.

**Out-of-vocabulary (OOV) words:** When `.transform()` sees a word that was not in the training vocabulary, there is no column for it, so that word is ignored. If all words in a test message are OOV, the row becomes all zeros.

**Data leakage:** If you fit the vectorizer on **both** training and test texts before evaluating, words that only appear in the test set get added to the vocabulary (and affect IDF weights). That leaks information from the test set into the feature representation, making validation/test metrics look optimistically better than real-world performance on truly unseen messages.

**Correct order of steps:**
1. Split the dataset into **train** and **test** (or train/validation/test) sets first.
2. Call `fit_transform` on the **training** text only.
3. Call `transform` on the **validation/test** text using that same fitted vectorizer.

---

### 7. Case Study: Text Representations on an African Language NLP Task

* **Study:** Nigusie, G., & Tegegne, T. (2022 / AfricanNLP 2025). *Amharic text complexity classification using supervised machine learning* (also discussed in African NLP literature on Amharic text classification; see Nigusie & Tegegne, 2022).
* **Task and Data:** The study focuses on **text complexity classification** in **Amharic** (an Ethiopian language with rich morphology and the Ge'ez script), classifying sentences into complex vs. non-complex levels across a corpus of Amharic sentences.
* **Representations Compared:** The authors compare traditional surface-level representations such as **Bag of Words (BoW)** alongside **word embeddings** (such as **Word2Vec** and **BERT** embeddings) paired with machine learning/deep learning classifiers.
* **Results and the "No Meaning" Limitation:** While frequency-based representations like BoW can capture basic surface word cues, they treat each vocabulary token as an isolated slot and do not naturally model semantic closeness between related or synonymous words—especially when rich Amharic inflectional and derivational forms split related concepts across different surface strings. Embedding representations capture distributed word relationships across contexts much more effectively than exact-match word counts alone, illustrating why BoW and TF-IDF struggle whenever tasks depend on underlying meaning rather than exact token overlap.

---

### References

* Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/
* Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
* Nigusie, G., & Tegegne, T. (2022). Amharic text complexity classification using supervised machine learning. *Proceedings of ICAST 2022* (also in *AfricaNLP* workshop proceedings).
* Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.

---

## Part B – Practical Report

### Task 2: Document-Term Matrix (`D1–D5`)

Columns are sorted alphabetically (32 features total across `D1–D5`):

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | **2** |
| **D2** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **D3** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **D4** | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **D5** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

### Task 7: `D2` Count vs. TF-IDF Table

| Word in `D2` | Count in `D2` | Document Frequency (`df` out of 5) | TF-IDF Weight (exact) | TF-IDF Weight (rounded 2 dp) |
| :--- | :---: | :---: | :---: | :---: |
| `call` | 1 | 1 | 0.3891 | 0.39 |
| `cash` | 1 | 1 | 0.3891 | 0.39 |
| `entry` | 1 | 1 | 0.3891 | 0.39 |
| `free` | 1 | 2 | 0.3140 | 0.31 |
| `now` | 1 | 2 | 0.3140 | 0.31 |
| `prizes` | 1 | 1 | 0.3891 | 0.39 |
| `today` | 1 | 2 | 0.3140 | 0.31 |
| `win` | 1 | 2 | 0.3140 | 0.31 |

### Task 8: Hand Calculations for `D2` (`cash` vs. `today`)

Using `CountVectorizer`'s default word token pattern (`\b\w\w+\b`), `D2` (`"Free entry: win cash prizes today. Call now."`) has **8** tokens: `['free', 'entry', 'win', 'cash', 'prizes', 'today', 'call', 'now']`, and $N = 5$ documents total.

* **`cash`:**
  * $\text{TF}(\text{cash}, \text{D2}) = \frac{1}{8} = 0.1250$
  * $\text{df}(\text{cash}) = 1$ (appears only in `D2`)
  * $\text{IDF}(\text{cash}) = \ln\!\left(\frac{5}{1}\right) = \ln(5) \approx 1.6094$
  * $\text{TF-IDF}(\text{cash}, \text{D2}) = 0.1250 \times 1.6094 \approx \mathbf{0.2012}$

* **`today`:**
  * $\text{TF}(\text{today}, \text{D2}) = \frac{1}{8} = 0.1250$
  * $\text{df}(\text{today}) = 2$ (appears in `D2` and `D3`)
  * $\text{IDF}(\text{today}) = \ln\!\left(\frac{5}{2}\right) = \ln(2.5) \approx 0.9163$
  * $\text{TF-IDF}(\text{today}, \text{D2}) = 0.1250 \times 0.9163 \approx \mathbf{0.1145}$

### Answers to Practical Questions

* **Task 1 (One-hot encoding):** For `free` and `cash` over `[cash, class, free, lunch, notes, win]`, the vectors are `[0, 0, 1, 0, 0, 0]` and `[1, 0, 0, 0, 0, 0]`. For `"win free cash"` (3 words), we get 3 separate one-hot vectors—one per word—so a longer sentence produces more vectors, whereas most ML models require a single fixed-length feature vector per input.
* **Task 2 (Bag of Words):** In `D1`, the cell for **`win`** has the value `2` because `"Win"` and `"WIN"` both appear in `D1` and lowercase to `"win"`. Transforming `"win free phone"` and `"phone free win"` produces the exact same vector (`1` for `free`, `phone`, and `win`, and `0` elsewhere), because bag of words ignores word order.
* **Task 4 (Preprocessing inside the vectorizer):** On `D1` with default settings, single-letter `"a"` and `"!"` are dropped, and `"Win"` and `"WIN"` merge into `"win"`. With `lowercase=False`, `"FREE"`, `"Reply"`, `"WIN"`, and `"Win"` become separate entries, and with `stop_words="english"`, `"to"` is removed. Across `D1–D5` with `stop_words="english"`, **`"not"` is removed** (`False`), which drops the negation in `D5` (`"The lecture was not good"`).
* **Task 5 (Limiting the vocabulary):** Out of 32 full vocabulary words, `min_df=2` keeps **7** words (`free`, `lecture`, `notes`, `now`, `the`, `today`, `win`) and `max_features=5` keeps **5** words (`free`, `lecture`, `the`, `today`, `win`). Both settings keep words that show up repeatedly across multiple documents rather than rare one-off words.
* **Task 6 (N-grams):** On `D5`, `ngram_range=(1, 2)` adds bigrams such as **`"not good"`** and **`"were great"`**, keeping the reviewer's mixed opinion (`not good` vs. `were great`) intact. Across `D1–D5`, vocabulary size grows from **32** (`(1, 1)`) to **67** (`(1, 2)`) and **98** (`(1, 3)`), roughly doubling and then tripling the number of features.
* **Task 7 (TF-IDF):** In `D2`, every word occurs once, so words appearing in fewer documents (`call`, `cash`, `entry`, `prizes`, with `df=1`, weight `0.39`) get higher TF-IDF weights than words appearing in two documents (`free`, `now`, `today`, `win`, with `df=2`, weight `0.31`). In `D1`, **`win`** gets the highest weight (`0.57`) because it occurs twice in `D1` (`count=2`), giving it a higher term frequency than the single-occurrence words in `D1`.
* **Task 8 (TF-IDF by hand):** My hand calculations give **`cash` = `0.2012`** and **`today` = `0.1145`**, ranking `cash` higher than `today`. Even though scikit-learn's smoothed, L2-normalized weights are larger (`0.39` vs. `0.31`), they rank the two words in the exact same order.
* **Task 10 (Fit and transform):** Fitting on `D1–D4` (26 features) and transforming `N1` leaves out-of-vocabulary words **`your`** and **`prize`**, while `N2` leaves out **`is`**, **`moved`**, and **`tomorrow`**. `"prize"` is unknown because `D2` has the plural `"prizes"` rather than singular `"prize"`, which lemmatization or stemming from Assignment 2 could normalize to a shared form.
* **Task 11 (Data leakage):** Fitting on `D1–D4` gives 26 features, while fitting on `D1–D4 + N1 + N2` gives 31 features. Five test words leaked into the vocabulary: **`is`**, **`moved`**, **`prize`**, **`tomorrow`**, and **`your`**.
* **Task 12 (No meaning):** Over `R1–R3`, the cosine similarity between every pair (`R1` and `R2`, `R1` and `R3`, `R2` and `R3`) is **`0.75`** (`3 / 4` overlapping words out of unit-norm vectors). `R1` (`tasty`) is **not** closer to `R2` (`delicious`) than to `R3` (`terrible`) because `tasty`, `delicious`, and `terrible` are separate columns with zero shared counts, so bag of words cannot tell that `tasty` and `delicious` mean something similar while `terrible` means the opposite.

---

### Recommended Vectorizer Settings for a Spam Detector

For a spam detector, I would use **`TfidfVectorizer` (or `CountVectorizer` with a baseline check) with `ngram_range=(1, 2)`**, **lowercasing**, and **careful vocabulary filtering (`min_df` / `max_features`)**, while **avoiding `stop_words="english"` and aggressive vocabulary pruning**. In my `D1–D5` outputs, spam cues like `"win"`, `"free"`, `"cash"`, `"claim"`, and `"now"` appear across spam messages (`D1` and `D2`), and bigrams help capture phrases like `"free prize"` or `"call now"`; at the same time, `min_df=2` kept useful repeated signals like `"free"`, `"now"`, and `"win"` while dropping noise. By contrast, `stop_words="english"` stripped `"not"` in `D5`, and in general default stopword lists can drop words that matter in context—plus fitting on `D1–D4 + N1 + N2` showed how easily test words (`"prize"`, `"your"`) leak into the vocabulary if you don't fit strictly on training data.