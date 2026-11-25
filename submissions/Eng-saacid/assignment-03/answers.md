# Assignment 3 – Vectorization

## Part A – Research

### 1. Text Vectorization

**Definition of Text Vectorization**

Text vectorization is the process of converting text into numerical vectors that Machine Learning models can understand and process. It is commonly used in NLP tasks such as spam detection, sentiment analysis, and text classification.

**Why do documents need the same number of features?**

Most Machine Learning models require a fixed number of numerical features for every document. This allows documents to be represented in a consistent matrix where each column represents the same feature.

**One-Hot Encoding**

One-hot encoding represents each word as a binary vector containing one `1` and the remaining values as `0`.

Example, using the vocabulary `[apple, banana, orange]`:

| Word | Vector |
|---|---|
| apple | [1, 0, 0] |
| banana | [0, 1, 0] |
| orange | [0, 0, 1] |

However, one-hot encoding represents individual words rather than complete documents. A sentence containing several words requires multiple vectors.

**Real-Life Example**

In spam detection, words such as "free" and "prize" can be converted into numerical features. A Machine Learning model uses these features to classify emails as spam or legitimate.

### 2. Bag of Words and Document-Term Matrix

**Bag of Words (BoW)** is a text representation method that counts how many times each vocabulary word appears in a document. It ignores the original order of words.

A Document-Term Matrix represents documents numerically:

- **Rows:** Individual documents.
- **Columns:** Unique words in the vocabulary.
- **Cells:** Frequency of each word in each document.

Example:

| Document | apple | banana | orange |
|---|---:|---:|---:|
| D1 | 2 | 1 | 0 |
| D2 | 0 | 1 | 2 |
| D3 | 1 | 0 | 1 |

It is called a "bag" because words are treated as an unordered collection.

**Advantages and Limitations**

- Losing word order causes little harm in topic classification, where word frequency can identify the subject.
- It causes problems in sentiment analysis because "not good" and "good" have different meanings, even though they contain similar words.

### 3. Vocabulary Size and Sparsity

**Vocabulary size** is the total number of unique words in a collection of documents.

**Sparsity** occurs when most cells in a document-term matrix contain zero because each document uses only a small portion of the complete vocabulary.

Real-world text collections can produce large matrices because they contain many unique words. Sparse formats such as CSR (Compressed Sparse Row) store non-zero values and their positions instead of storing every zero, reducing memory usage.

**Settings that control vocabulary size:**

| Setting | Function |
|---|---|
| `min_df` | Removes terms appearing in fewer than the specified number of documents. |
| `max_df` | Removes terms appearing in too many documents. |
| `max_features` | Limits the vocabulary to the most frequent terms. |

**Preprocessing effects:**

- Lowercasing converts uppercase and lowercase versions into the same form.
- Stopword removal removes common words such as "the" and "is".
- Stemming reduces related word forms to a common stem.

These techniques reduce unnecessary vocabulary features and can improve text processing.

### 4. N-grams

N-grams are sequences of consecutive words used to preserve some word order in text.

| Type | Description | Example |
|---|---|---|
| Unigram | One word | `machine` |
| Bigram | Two consecutive words | `machine learning` |
| Trigram | Three consecutive words | `natural language processing` |

**Example:**

The bigram `not good` preserves the negative meaning of a phrase that individual unigrams may miss.

**Cost of Larger N-grams**

Larger n-grams increase vocabulary size, memory requirements, and processing time. They can also create many rare features, making the model more complex.

### 5. TF-IDF

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It measures how important a word is in a document compared with the entire collection.

The idea of inverse document frequency was developed by Karen Spärck Jones in her 1972 research on term specificity in information retrieval.

**Textbook Formulas**



Term Frequency:

```text
TF     = (times the word appears in the document) / (words in the document)
````

Inverse Document Frequency:

```text
IDF    = log(N / df)      N = number of documents, df = documents containing the word
````

TF-IDF:

```text
TF-IDF = TF × IDF
````

Where:

- \(t\) = Term (word).
- \(d\) = Document.
- \(N\) = Total number of documents.
- \(df(t)\) = Number of documents containing the term.

**Why do rare words get higher weights?**

Rare words receive higher IDF values because they appear in fewer documents and can help distinguish one document from another. Common words receive lower weights because they provide less information.

**Scikit-learn Differences**

Scikit-learn's `TfidfVectorizer` uses smoothing and normalization by default:

```text
IDF(t) = log((1 + N) / (1 + df(t))) + 1
```

It also applies L2 normalization by default, making each non-zero document vector have a unit length.

**Why are TF-IDF values weights, not probabilities?**

TF-IDF values represent the relative importance of words, not the probability of an event. They do not have to sum to 1 and are not probabilities.

### 6. Fit, Transform and Data Leakage

**Difference between Fit, Transform and Fit_Transform**

| Method | Function |
|---|---|
| `fit()` | Learns the vocabulary and necessary statistics from training documents. |
| `transform()` | Converts documents using the learned vocabulary. |
| `fit_transform()` | Learns the vocabulary and transforms the documents in one operation. |

**Out-of-Vocabulary (OOV) Words**

OOV words are words that were not included in the vocabulary during training. When new documents are transformed, these words are ignored by standard `CountVectorizer` and `TfidfVectorizer`.

**Data Leakage**

Data leakage occurs when information from the test set is used during training or preprocessing. Fitting a vectorizer on both training and test documents allows test vocabulary information to influence the training process, making evaluation less reliable.

**Correct Order of Steps**

1. Split the dataset into training and testing sets.
2. Fit the vectorizer on training data only.
3. Transform the training data.
4. Transform the test data using the same fitted vectorizer.
5. Train and evaluate the model.

### 7. Case Study: Text Representation for African Language Sentiment Analysis

**Study:** *Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages* (Aryal et al., 2023).

**Task and Dataset:** The researchers investigated sentiment classification using the AfriSenti-SemEval datasets. The study covered 12 African languages, including Hausa, Yoruba, Igbo, Amharic, and Swahili. The datasets contained social media posts labelled as positive, negative, or neutral.

**Representations Compared:**
- **Delta TF-IDF:** A statistical text representation that assigns importance to words based on their distribution across sentiment classes.
- **Transformer representations:** Pre-trained language models such as AfriBERTa, AfroXLMR, and AfroLM, which capture contextual information.

**Results:** Transformer models performed better on average. However, Delta TF-IDF remained competitive and sometimes achieved better results for particular languages. For example, the SVM model using Delta TF-IDF achieved a weighted F1-score of 0.75 for Hausa, compared with 0.78 for AfroXLMR. The study also found that traditional TF-IDF-based methods required significantly less training data.

**What This Shows:** Bag of Words and TF-IDF represent text using word-based numerical features, but they do not fully understand context or meaning. TF-IDF improves on basic word counts by giving more importance to informative words. Transformer representations can capture contextual meaning, but they generally require more computational resources and training data.



## References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

2. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/EB026526

3. Scikit-learn. *CountVectorizer Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

4. Scikit-learn. *TfidfVectorizer Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

5. Scikit-learn. *Feature Extraction: Text Feature Extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html

6. Rudkowsky, E., Haselmayer, M., Wastian, M., Jenny, M., Emrich, Š., & Sedlmair, M. (2018). More than Bags of Words: Sentiment Analysis with Word Embeddings. *Communication Methods and Measures, 12*(2–3), 140–157. https://doi.org/10.1080/19312458.2018.1455817
   
7. Aryal, S. K., Prioleau, H., Shah, U., & Acharya, S. (2023). *Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages*. The Journal of Computing Sciences in Colleges, 39(3), 327–335. https://ccsc-eastern.github.io/files/ccsce23/EA2023.pdf