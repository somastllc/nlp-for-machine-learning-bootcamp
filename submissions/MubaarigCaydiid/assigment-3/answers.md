# Assignment 3 – Vectorization

## Part A – Research

### 1. What Is Text Vectorization?

Text vectorization is the process of converting text into numerical representations that machine learning algorithms can understand and process.

Machine learning models generally require numerical input rather than raw text. Therefore, vectorization transforms words, sentences, or documents into numerical vectors. Each document is represented using a consistent set of features so that the resulting vectors have the same dimensions.

**One-Hot Encoding** is a basic vectorization technique that represents each word using a binary vector. Each position represents a word in the vocabulary. A value of `1` indicates the presence of that word, while `0` indicates its absence.

**Example:** Consider the vocabulary `[class, free, notes, prize]`.

| Word | `class` | `free` | `notes` | `prize` |
|---|---:|---:|---:|---:|
| `free` | 0 | 1 | 0 | 0 |
| `prize` | 0 | 0 | 0 | 1 |

One-hot encoding represents each word as a separate vector. Therefore, a sentence becomes a list of word vectors. Since sentences can have different numbers of words, this representation can produce lists of different lengths. Bag of Words provides a fixed-length vector for each document.

**Real-life example:** In an email spam detection system, words such as `free`, `win`, and `prize` can be converted into numerical features. A machine learning model can use these features to identify potentially unwanted emails.

### 2. Bag of Words (BoW)

Bag of Words (BoW) is a text representation technique that converts documents into numerical vectors based on the frequency of words in a vocabulary.

It creates a **Document-Term Matrix (DTM)**, where:

- **Rows** represent individual documents.
- **Columns** represent unique words in the vocabulary.
- **Cells** contain the frequency of each word in a document.

It is called a "bag" because it treats a document as a collection of words without considering their original order or grammatical structure.

**When losing word order is acceptable:** In basic spam detection, the frequency of words such as `offer`, `free`, and `winner` may provide useful information even if their order is ignored.

**When losing word order is harmful:** In sentiment analysis, the phrases `not good` and `good` have different meanings. A basic BoW representation may fail to capture this difference because it does not preserve word order or context.

### 3. Vocabulary Size and Sparsity

**Vocabulary size** refers to the total number of unique words identified across a collection of documents.

A Document-Term Matrix can become very large and sparse because:

- A collection may contain thousands of unique words.
- Each document usually contains only a small portion of the complete vocabulary.
- Most documents do not contain most words in the vocabulary.

This results in many zero values in the matrix. Such a matrix is called a **sparse matrix**.

Sparse matrix storage is useful because it stores non-zero values and their positions instead of storing every zero. This reduces memory usage and can improve computational efficiency.

Scikit-learn provides several parameters for controlling vocabulary size:

- **`min_df`**: Removes words that appear in fewer than the specified number of documents.
- **`max_df`**: Removes words that appear in more than the specified proportion or number of documents.
- **`max_features`**: Limits the vocabulary to a maximum number of features, selected according to term frequency.

Text preprocessing also affects vocabulary size:

- **Lowercasing** combines uppercase and lowercase versions of the same word.
- **Stopword removal** removes common words that may provide little useful information for a particular task.
- **Stemming** reduces related word forms to a common stem, potentially reducing the number of unique features.

### 4. N-Grams

An N-gram is a continuous sequence of *n* items, usually words, extracted from a text.

The main types include:

- **Unigram (1-gram):** A single word, such as `machine`.
- **Bigram (2-gram):** A sequence of two consecutive words, such as `machine learning`.
- **Trigram (3-gram):** A sequence of three consecutive words, such as `natural language processing`.

Unlike a basic unigram representation, n-grams can preserve some information about word order.

**Example:** The bigrams `not good` and `very good` represent different expressions and can help a sentiment analysis model distinguish between them.

However, increasing the n-gram range also increases the number of possible features. This can produce a larger vocabulary, greater memory requirements, and a more sparse feature matrix. Therefore, selecting an appropriate n-gram range depends on the task and available data.

### 5. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF is a statistical weighting technique used to measure how important a word is within a document compared with a collection of documents.

It combines two measurements:

- **Term Frequency (TF):** How often a word appears in a particular document.
- **Inverse Document Frequency (IDF):** How rare or informative a word is across the document collection.
- **TF-IDF:** The product of TF and IDF.

#### 5.1 Term Frequency (TF)

TF measures how frequently a word appears in a particular document.

**Formula:**

```text
TF(t, d) = (Number of times term t appears in document d)
           / (Total number of words in document d)
```

Where:

- `t` = Term (word).
- `d` = Document.
- `f(t, d)` = Number of times term `t` appears in document `d`.

The formula can also be written as:

```text
TF(t, d) = f(t, d) / Total words in d
```

**Example:** Consider the document:

```text
I love Python Python
```

The document contains 4 words, and `Python` appears 2 times.

```text
TF(Python, d) = 2 / 4
TF(Python, d) = 0.5
```

#### 5.2 Inverse Document Frequency (IDF)

IDF measures how rare or informative a word is across the entire document collection. A word that appears in many documents receives a lower IDF score, while a word that appears in fewer documents receives a higher score.

**Textbook formula:**

```text
IDF(t) = log(N / df(t))
```

Where:

- `N` = Total number of documents.
- `df(t)` = Number of documents containing term `t` at least once.
- `log` = Logarithm.

**Example:** If a collection contains 5 documents and the word `cash` appears in 2 documents:

```text
IDF(cash) = log(5 / 2)
IDF(cash) ≈ 0.916
```

#### 5.3 TF-IDF Calculation

TF-IDF combines Term Frequency and Inverse Document Frequency by multiplying their values.

**Formula:**

```text
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Example:** Using the previous calculations:

```text
TF-IDF(cash, d) = 0.2 × 0.916
TF-IDF(cash, d) ≈ 0.183
```

This means that the TF-IDF weight of `cash` in this document is approximately **0.183**.

#### Scikit-learn Implementation

Scikit-learn's `TfidfVectorizer` uses a smoothed IDF formula by default:

```text
IDF(t) = log((1 + N) / (1 + df(t))) + 1
```

The `+1` smoothing prevents zero IDF values. By default, `TfidfVectorizer` also applies L2 normalization to the resulting document vectors.

**Important:** Scikit-learn's default TF calculation uses raw term frequency rather than the document-length-normalized TF formula shown in the textbook example. Therefore, manually calculated TF-IDF values may differ from `TfidfVectorizer` results.

TF-IDF values are **not probabilities**. They are numerical weights that represent the relative importance of terms within a document collection.

The statistical foundations of inverse document frequency were discussed by Karen Spärck Jones in her influential 1972 paper, *A Statistical Interpretation of Term Specificity and Its Application in Retrieval*.

### 6. `fit()`, `transform()`, `fit_transform()`, and Data Leakage

Scikit-learn vectorizers provide three important methods:

- **`fit()`**: Learns the vocabulary and other required information from the training text.
- **`transform()`**: Converts text into numerical vectors using the vocabulary learned during fitting.
- **`fit_transform()`**: Performs both operations: learning the vocabulary and transforming the supplied text.

**Out-of-Vocabulary (OOV) words** are words that were not included in the vocabulary learned during the fitting process. When a vectorizer transforms new text, words outside its vocabulary are ignored by standard `CountVectorizer` and `TfidfVectorizer` implementations.

**Data leakage** occurs when information from the test dataset is used during model training or preprocessing. For example, fitting a vectorizer on both training and test data allows the vocabulary to include information from the test set.

This can make model evaluation unreliable and may produce overly optimistic performance results.

**Correct workflow:**

1. Split the dataset into training and testing sets.
2. Fit the vectorizer using only the training data.
3. Transform the training data.
4. Transform the test data using the same fitted vectorizer.
5. Train and evaluate the machine learning model.

This ensures that the test data remains unseen during the fitting process.

### 7. Research Case Study: Text Representation for African Language Sentiment Classification

Aryal, Prioleau, Shah, and Acharya (2023) conducted a study titled *Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages*.

**Research objective:** The study investigated how traditional term-frequency-based representations compare with Transformer-based language models when performing sentiment classification on African languages.

**Dataset:** The researchers used social media datasets covering several African languages, including Hausa, Yoruba, Igbo, Swahili, Amharic, and Nigerian Pidgin. The data included sentiment labels such as positive, negative, and neutral.

**Representations and methods:** The study evaluated a traditional feature-based approach using Delta TF-IDF alongside conventional machine learning algorithms. It also compared these results with Transformer-based models, including AfriBERTa and AfroXLMR.

Delta TF-IDF is an extension of traditional TF-IDF that considers how strongly terms are associated with different classes.

**Key findings:** The researchers found that Transformer models achieved better average performance overall. However, Delta TF-IDF remained competitive and could perform well in settings where pretrained language models or sufficient training resources were unavailable.

**Limitations of BoW and TF-IDF:** Traditional BoW and TF-IDF representations mainly rely on word occurrence and frequency. They do not naturally understand the meaning of words, relationships between words, or context. For example, they may treat `excellent` and `terrible` as unrelated features without understanding their opposite sentiment.

**Conclusion:** The study demonstrates that traditional vectorization techniques remain useful for low-resource language tasks, particularly when computational resources and training data are limited. However, contextual language models can capture richer linguistic information.

---

## References

1. Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526
2. Scikit-learn Developers. (n.d.). *Feature extraction: Text feature extraction*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/feature_extraction.html
3. Aryal, S. K., Prioleau, H., Shah, U., & Acharya, S. (2023). Term frequency features vs transformers: A comparison for sentiment classification of African languages. *Journal of Computing Sciences in Colleges, 39*(3), 327–335. https://www.ccsc.org/publications/journals/EA2023.pdf