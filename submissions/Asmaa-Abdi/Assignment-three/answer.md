# Assignment-3 Vectorization

## Part A

1. text vectorization is the process of converting textual data — like sentences or documents — into numerical format so that it can be understood and processed by machine learning algorithms. Machines work with numbers, and not raw text, vectorization acts as a bridge between human language and machine learning models.

Because most Machine Learning models require a uniform feature matrix or vector space, every document must be represented with the exact same number of numerical features in a consistent column order so the model can compute weights, distances, and dot products reliably across all samples.

2. Computers need numbers, not words. To process language, we first turn words into numbers. One of the earliest ways to do this is one-hot encoding represents each word in your vocabulary with a vector—a list of numbers—whose length matches the vocabulary size. The main application of one-hot encoding is to represent symbols with vectors without any prior bias. 'book', 'apple', 'library' are symbolic representations without any measurable values.

A single word's one-hot vector is not enough to represent an entire document because a document contains multiple words, repeated terms, and contextual relationships that one isolated word vector cannot capture. In addition, distinct one-hot vectors are mutually orthogonal, meaning they do not naturally measure semantic similarity between related words (such as synonyms) or preserve word order.

For example, in a customer support ticket reading "Payment failed twice", a single one-hot word vector only flags one isolated token rather than the full complaint, and plain one-hot vectors treat "failed" and "declined" as unrelated even though both describe a billing problem.

Bag of Words (BoW) · Is one of the most fundamental techniques in natural language processing used to convert textual data into numerical format. · It represents a piece of text — such as a sentence or document — as a “bag” of its individual words, disregarding grammar, word order, and punctuation, but keeping track of word frequency.

A document–term matrix is a table that turns language into numerical features: each row represents a document, each column represents a word or phrase under defined processing rules, and each cell records presence, a count or a weight. This representation makes text collections searchable and comparable. It does not, by itself, preserve who said what, the order of words, negation or the truth of a claim.

Task where losing word order causes little harm: Topic classification (such as sorting news articles into Sports, Tech, or Finance), because topic keywords alone (like "goal", "market", or "software") carry enough signal to identify the category even when word order is lost.

3. Vocabulary size is the number of distinct tokens a software system can recognize or generate. In artificial intelligence and natural language processing, it usually describes a tokenizer's inventory, not the number of complete dictionary words a person knows.

sparsity is a matrix with a significant number of zero elements, making it memory-inefficient when stored conventionally. Unlike dense matrices, which store every element explicitly, sparse matrices use specialized representations to save storage and enhance computational efficiency.

Real-world document-term matrices become very large and mostly zero because each individual document uses only a small fraction of all words that appear across the whole corpus, while rare terms, proper names, and spelling variants keep adding new columns. Sparse matrices are stored efficiently using formats like Compressed Sparse Row (CSR) or Compressed Sparse Column (CSC), which store only non-zero values and their row/column index pointers instead of storing every zero.

Vocabulary size and sparsity also change with vectorizer parameters and preprocessing choices:

* min_df excludes rare terms that appear in fewer than a minimum number or fraction of documents, shrinking the vocabulary and removing noisy one-off terms.
* max_df excludes overly common terms that appear in more than a maximum fraction of documents, filtering out corpus-wide words with low discriminative power.
* max_features caps the vocabulary at the top most frequent terms across the corpus, keeping matrix width bounded.
* LowerCasing Text often has a variety of capitalization reflecting the beginning of sentences or proper nouns emphasis (such as `"Win"` and `"win"`) into a single column, reducing vocabulary size.
* Stop words are a set of commonly used words in a language like “a”, “the”, “is”, “are” and etc in English.
* Stemming is the process of eliminating affixes (suffixes, prefixes) from a word in order to obtain a word stem.

4. An N-gram is a contiguous sequence of n items, usually words, but also characters, syllables, phonemes, or even DNA base pairs, taken from a piece of text or speech.

A unigram is a single word, the most basic unit of text. Unigrams for the sample: (“Follow,” “good,” “SEO,” “practices”). You take one word at a time.

A bigram is two adjacent words. Bigrams for the sample: (“Follow good,” “good SEO,” “SEO practices”). You take two words at a time, sliding one word forward each step.

A trigram is three adjacent words. Trigrams for the sample: (“Follow good SEO,” “good SEO practices”).

However, using larger n-grams comes with significant computational and statistical costs. As $n$ increases, the number of unique word combinations grows rapidly, causing vocabulary size, memory usage, and training time to expand sharply while making the document-term matrix much sparser because most multi-word sequences occur only rarely.

5. TF-IDF, short for Term Frequency-Inverse Document Frequency, is a statistical measure used in natural language processing (NLP) and information retrieval to evaluate the importance of a word to a document within a collection of documents (called a corpus).

Inverse Document Frequency(IDF) comes from Karen Spärck Jones (1972).

textbook formula : TF-IDF(t,d) = TF(t,d)*IDF(t) || TF(t,d) = numer of term apears in document d / numer of term in document d ||| IDF (ti) = log N / n1

* **Why rarer words get higher weights:** When a term appears in only a few documents (low document frequency), the inverse document frequency ratio becomes larger, giving higher weight to more distinctive terms than to words appearing across almost every document.
* **How scikit-learn's `TfidfVectorizer` differs from the bare textbook formula:**
  * **Smoothing (`smooth_idf=True`):** Adds `1` to both numerator and denominator and adds `1` outside the logarithm, preventing division by zero and ensuring terms present in all documents still retain a positive weight.
  * **Normalization (`norm='l2'`):** Scales each document vector to unit L2 norm so longer documents do not automatically have larger vector magnitudes than shorter ones.
* **Why TF-IDF scores are weights, not probabilities:** Before normalization, TF-IDF values are not bounded between `0` and `1`, and across terms or documents they do not sum to `1` to form a probability distribution—they are heuristic importance weights rather than probabilities.

6. fit: Learn and estimate the parameters of the transformation

transform: Apply the learned transformation to new data

fit_transform: Learn the parameters and apply the transformation to new data

Every NLP model is trained on a fixed vocabulary derived from the training dataset. If a word outside this vocabulary shows up during testing or deployment, it’s classified as out-of-vocabulary (OOV).

Data leakage occurs when a model unintentionally uses information that would not be available during real-world predictions. This leads to overly optimistic performance during validation, as the model learns patterns it should not have access to, resulting in poor generalization in practical scenarios.

* **How a vectorizer handles OOV words during `transform`:** Standard bag-of-words or TF-IDF vectorizers ignore (drop) out-of-vocabulary words encountered during `transform` because those words were not present in the vocabulary learned during `fit` on the training data.
* **Correct order of steps to prevent data leakage:**
  1. Split the dataset into training and test sets first.
  2. Call `fit` (or `fit_transform`) only on the training set so the vocabulary and weights come strictly from training data.
  3. Call `transform` on the test set using that fitted vectorizer.

7. **Case Study: Izindaba-Tindzaba (isiZulu and Siswati News Categorisation)**
* **Task:** News topic classification (categorising news articles and headlines into topical categories) for long and short texts in isiZulu and Siswati.
* **Data:** Curated and annotated isiZulu and Siswati news datasets (long articles and short headlines) built for low-resource South African languages.
* **Representations compared:** Bag of Words (BoW), TF-IDF, and Word2vec word embeddings evaluated across machine learning and deep learning models (including Logistic Regression, Naive Bayes, XGBoost, and LSTM).
* **Results:** Word2vec embeddings generally outperformed Bag of Words and TF-IDF across experiments, showing the advantage of dense distributed representations over pure count/frequency features on these classification tasks.

**How this shows BoW and TF-IDF lack meaning:** Bag of Words and TF-IDF represent text using discrete word counts or frequency-based weights without capturing semantic similarity or context between words. Because they treat terms as independent count or weight features rather than encoding contextual meaning, they miss deeper semantic relationships that embedding representations capture—especially in morphologically rich languages.

**Reference:** Madodonga, A., Marivate, V., & Adendorff, M. (2023). *Izindaba-Tindzaba: Machine learning news categorisation for Long and Short Text for isiZulu and Siswati*. Journal of the Digital Humanities Association of Southern Africa, 4(01).

---

## Part B

* **One-Hot Encoding & Bag of Words:** Each word is represented as a fixed-length vector matching vocabulary size, while `CountVectorizer` produces a document–term matrix where rows are documents, columns are vocabulary terms, and cells store counts. `"win free phone"` and `"phone free win"` produce the exact same count vector because Bag of Words ignores word order.
* **Vocabulary Size & Preprocessing:** `CountVectorizer` lowercases text by default (merging `"Win"` and `"win"`), whereas `lowercase=False` keeps case variants separate. Using `stop_words="english"` removes common English function words, `min_df=2` keeps only terms appearing in at least two documents, and `max_features=5` restricts the vocabulary to the top five features.
* **N-Grams:** Expanding `ngram_range` from `(1, 1)` to `(1, 2)` or `(1, 3)` adds adjacent word sequences (bigrams and trigrams) that preserve local word order and context (such as `"not good"`), while increasing feature count and sparsity.
* **TF-IDF & Sparsity:** `TfidfVectorizer` weights terms using term frequency and inverse document rarity (with smoothing and L2 normalization by default). In `count_matrix`, most entries are zero (`total - nnz` zero cells versus `nnz` non-zero cells), demonstrating sparsity because each document uses only a small fraction of the full vocabulary.
* **Fit vs. Transform, Data Leakage & Meaning:** Fitting `CountVectorizer` only on `train` and calling `transform` on `N1` and `N2` avoids data leakage and drops unseen test words, whereas fitting on `train + test` leaks test vocabulary into training. For `"the food was tasty"`, `"the food was delicious"`, and `"the food was terrible"`, count vectors treat all three descriptive adjectives as separate, unrelated features with no semantic similarity encoded.
