# Text Vectorization – Assignment

## 1. Text Vectorization and One-Hot Encoding

Text vectorization is the process of converting words, sentences, or documents into numerical vectors (lists or arrays of numbers) so machine learning models and algorithms can process human language.

Most traditional machine learning models expect a fixed-size numerical feature matrix.

### One-Hot Encoding

One-Hot Encoding turns each unique word in a vocabulary into a binary vector of length equal to the vocabulary size. Only the index assigned to that word is set to 1, and every other position is 0.

### Why Is One-Hot Encoding Alone Not Enough for a Document?

One-Hot Encoding alone is not enough because it represents each word separately. It creates large, sparse vectors and does not capture word frequency, importance, or meaning.

**Example:**

Consider a spam message:

`win free cash now`

Each word is encoded separately. However, One-Hot Encoding does not directly represent the complete document or show which words are more important.

---

## 2. Bag of Words (BoW)

Bag of Words converts text into numerical vectors by counting word frequencies while ignoring grammar and word order.

### Document-Term Matrix

A document-term matrix is a table where:

- **Rows:** Documents.
- **Columns:** Unique words.
- **Cells:** Number of times each word appears in a document.

It is called a bag because words are treated as if they were in a bag, without considering their order or grammar.

### Examples

- Losing word order causes little harm in spam detection.
- Losing word order causes real problems in sentiment analysis because "I love this" and "I hate this" have different meanings.

---

## 3. Vocabulary Size and Sparsity

**Vocabulary size:** The total number of unique words in a collection of documents.

**Sparsity:** The percentage of cells in a document-term matrix that contain zero.

Real document-term matrices become large and mostly zero because documents use only a small fraction of all possible vocabulary words.

Sparse matrices store only non-zero values and their positions, saving memory.

### Vectorizer Settings and Preprocessing

- `min_df`: Removes words appearing in too few documents.
- `max_df`: Removes words appearing in too many documents.
- `max_features`: Limits the vocabulary to a maximum number of features.
- **Lowercasing:** Treats `Apple` and `apple` as the same word.
- **Stopword removal:** Removes common words like `the`, `is`, and `and`.
- **Stemming:** Reduces related words to a common stem, e.g., `playing` and `played` → `play`. The resulting stem is not always a valid dictionary word.

**Example of sparsity:**

If 90% of the cells in a document-term matrix contain zeros, its sparsity is 90% and its density is 10%.

---

## 4. N-Grams

N-grams are groups of consecutive words extracted from text. They help machine learning models capture word order and relationships between neighboring words.

### Types of N-Grams

- **Unigram:** Treats each word separately. It does not preserve word order.
- **Bigram:** Captures groups of two consecutive words.
- **Trigram:** Captures groups of three consecutive words.

### Example

Consider these sentences:

- The food is good.
- The food is not good.

Unigrams see one word at a time, while bigrams see two consecutive words.

For example, the bigrams in "The food is not good" are:

- The food
- food is
- is not
- not good

The bigram `not good` helps the model recognize negative meaning.

### Limitations of Larger N-Grams

Larger n-grams create more features, use more memory, increase sparsity, and can cause overfitting because many word combinations are rare.

---

## 5. TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF measures how important a word is in a document compared with all documents.

- **TF (Term Frequency):** Measures how often a word appears in a document.
- **IDF (Inverse Document Frequency):** Measures how rare a word is across all documents.

The IDF idea was introduced by **Karen Spärck Jones in 1972** to identify words that help distinguish documents.

### Textbook Formula
## TF-IDF Formulas

**1. Term Frequency (TF)**

Measures how often a word appears in a document.

TF = (Number of times the word appears) / (Total words in the document)

**2. Inverse Document Frequency (IDF)**

Measures how rare a word is across all documents.

IDF = log(N / df)

- N = Total number of documents
- df = Number of documents containing the word

**3. TF-IDF**

Measures the importance of a word in a document.

TF-IDF = TF × IDF

Rare words get higher weights because they appear in fewer documents and help identify specific documents.

### Scikit-learn Differences

Scikit-learn's `TfidfVectorizer` differs from the textbook formula in several ways:

- **Smoothing:** Adds 1 to the numerator and denominator to avoid division by zero.

  \[
  IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
  \]

- **Term Frequency:** Uses raw term frequency by default.
- **Normalization:** Applies L2 normalization by default, scaling each non-zero document vector so its Euclidean length is 1.

### Why Weights, Not Probabilities?

TF-IDF measures word importance, not the likelihood of an event. Its values do not need to add up to 1 because they represent numerical importance scores, not probabilities.

---

## 6. fit, transform, and fit_transform

- **`fit()`:** Learns the vocabulary and word statistics from training text.
- **`transform()`:** Converts text into numerical vectors using the learned vocabulary.
- **`fit_transform()`:** Performs both `fit()` and `transform()` at the same time.

### Out-of-Vocabulary (OOV) Words

Out-of-vocabulary words are words that were not seen during `fit()`. They are ignored during transformation because they are not in the learned vocabulary.

### Data Leakage

Data leakage happens when we fit the vectorizer using both training and testing data.

This allows information from the test data to influence the vocabulary and IDF values, making evaluation results look better than they really are.

The test data is no longer completely independent of the training process.

### Correct Order of Steps

1. Split the data into training and testing sets.
2. Apply `fit_transform()` on training data.
3. Apply `transform()` on testing data.
4. Train the model using training vectors.
5. Evaluate using testing vectors.

**Note:** Using a Scikit-learn Pipeline can also help prevent data leakage when performing cross-validation.

---


## Case Study: TF-IDF vs Transformers in African Languages

**Research paper:** Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages (2023).

**Authors:** Saurav K. Aryal, Howard Prioleau, Ujjawal Shah, and Sameer Acharya.

### 1. Task
The researchers compared traditional machine learning and transformer approaches for sentiment analysis (positive, negative, and neutral) in African languages.

### 2. Dataset
They used the AfriSenti-SemEval dataset, which contains 14 annotated African-language datasets, including Hausa, Yoruba, Swahili, and Igbo.

### 3. Text Representations
- **Delta TF-IDF:** Represents words using their statistical importance and sentiment-class differences.
- **Transformers:** Learn contextual representations of text.

### 4. Results
Delta TF-IDF achieved promising and competitive results while requiring significantly less training data than transformer models.

### 5. Limitation of TF-IDF
TF-IDF measures word importance but does not inherently understand meaning or context.

For example:
- "This product is good."
- "This product is not good."

Both sentences contain almost the same words, but their meanings are opposite. TF-IDF does not inherently understand how "not" changes the meaning.

Transformers learn contextual relationships between words, helping models distinguish these sentences.

### 6. Conclusion
Bag of Words counts word occurrences, while TF-IDF measures word importance. Word embeddings learn word relationships, and contextual embeddings represent words based on surrounding context.

**Key lesson:** Knowing which words appear in a sentence is not the same as understanding what the sentence means.

### Reference
Aryal et al. (2023). *Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages.*

[Read the research paper](https://ccsc-eastern.github.io/files/ccsce23/EA2023.pdf)
