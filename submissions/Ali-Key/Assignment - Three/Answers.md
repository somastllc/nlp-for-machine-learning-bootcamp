# Assignment Three: Vectorization

**Author:** Ali Omar Abdi  
**Subject:** Natural Language Processing (NLP)

## Part A – Research

### 1. What is Text Vectorization?

Text vectorization is the process of changing text into numbers so that a computer can process it.

Machine Learning models usually work with numerical data. Therefore, we must convert documents into numbers before giving them to a model.

Most models also need every document to have the same number of features. A **feature** is one piece of information used by a model. Using the same vocabulary gives every document the same feature positions.

**One-hot encoding**

One-hot encoding represents each word with a vector containing one `1` and zeros in the other positions.

For example, consider this vocabulary:

`[cat, dog, bird]`

| Word | One-hot vector |
|---|---|
| cat | `[1, 0, 0]` |
| dog | `[0, 1, 0]` |
| bird | `[0, 0, 1]` |

One-hot encoding represents individual words. However, it does not represent a complete document by itself. A document contains several words, and each word needs its own representation.

**Real-life example**

Imagine a library system that stores book titles. The computer needs numerical information to search or classify the titles. One-hot encoding can identify individual words, but it cannot directly explain the full meaning of a title.

### 2. Bag of Words and the Document-Term Matrix

**Bag of Words (BoW)** is a method that represents a document by counting how many times each word appears.

It is called a "bag" because it treats words as if they are placed in a bag. Their original order is not preserved.

A **document-term matrix (DTM)** is a table that shows the words found in a collection of documents.

| Part | Meaning |
|---|---|
| Rows | Documents |
| Columns | Unique words |
| Cells | Number of times each word appears |

**Example**

Consider these documents:

- Document 1: `I like reading books`
- Document 2: `I like reading novels`

The matrix uses one shared vocabulary for both documents. Each number shows how many times a word appears in a document.

**When is BoW useful?**

BoW can work well for news classification. Words such as `football`, `player`, and `match` can help identify sports articles.

**When can BoW cause problems?**

BoW does not preserve word order.

For example:

- `The service was good.`
- `The service was not good.`

These sentences have different meanings. A basic BoW model may struggle to identify this difference because it treats the words separately.

### 3. Vocabulary Size and Sparsity

**Vocabulary size** means the number of unique words in a collection of documents.

For example, if a collection contains 5,000 different words, its vocabulary size is 5,000.

**Sparsity** means that most values in a matrix are zero.

A document usually contains only a small number of the words in the complete vocabulary. Therefore, most cells in a document-term matrix contain zero.

**Why are sparse matrices useful?**

Storing every zero uses unnecessary memory. Sparse matrices save space by storing non-zero values and information about their positions.

The SciPy library supports different sparse matrix formats for efficient storage and calculations (SciPy, n.d.).

**Settings that change vocabulary size**

| Setting | Explanation |
|---|---|
| `min_df` | Removes words that appear in very few documents. |
| `max_df` | Removes words that appear in too many documents. |
| `max_features` | Limits the number of words selected. |
| Lowercasing | Changes words to lowercase, so `Book` and `book` are treated as the same word. |
| Stopword removal | Removes selected common words, such as `the` and `is`. |
| Stemming | Reduces related word forms to a common stem. |

These settings can reduce vocabulary size and memory use. However, preprocessing must be done carefully. For example, removing the word `not` can change the meaning of a sentence.

### 4. What are N-grams?

An **n-gram** is a sequence of consecutive words. N-grams help preserve some word order.

There are three common types:

| Type | Meaning | Example |
|---|---|---|
| Unigram | One word | `good` |
| Bigram | Two words | `not good` |
| Trigram | Three words | `was not good` |

**Example**

Consider this sentence:

`The service was not good.`

- Unigrams: `The`, `service`, `was`, `not`, `good`
- Bigrams: `The service`, `service was`, `was not`, `not good`
- Trigrams: `The service was`, `service was not`, `was not good`

A bigram can help identify the difference between `not good` and `very good`.

**What is the disadvantage of larger n-grams?**

Larger n-grams create more possible features. This can increase vocabulary size, memory use, and processing time.

They may also produce many features that appear only once. This makes it harder for a model to learn useful patterns from a small dataset (Manning et al., 2008).

### 5. What is TF-IDF?

**TF-IDF** means Term Frequency–Inverse Document Frequency. It gives a weight to each word based on how often it appears in a document and how common it is across all documents.

It has two parts:

- **Term Frequency (TF):** How often a word appears in one document.
- **Inverse Document Frequency (IDF):** How uncommon the word is across the document collection.

The textbook formula is:

$$TF\text{-}IDF(t,d)=TF(t,d)\times IDF(t)$$

The IDF formula is:

$$IDF(t)=\log\left(\frac{N}{df(t)}\right)$$

Where:

- \(t\) = the word.
- \(d\) = the document.
- \(N\) = total number of documents.
- \(df(t)\) = number of documents containing the word.

**Why do rare words get higher weights?**

A word that appears in almost every document does not help distinguish one document from another.

A rare word may provide more useful information. Therefore, TF-IDF gives rare words higher IDF values.

**Where did IDF come from?**

The idea of inverse document frequency is associated with Karen Spärck Jones's 1972 research paper. Her work explained how the number of documents containing a word can help estimate its importance in information retrieval (Spärck Jones, 1972).

**How is scikit-learn different?**

Scikit-learn's `TfidfVectorizer` uses a smoothed IDF formula by default:

$$IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1$$


The added `1` values help avoid calculation problems.

Scikit-learn also uses L2 normalization by default. This adjusts each document vector so that its length is 1, when the vector is non-zero (scikit-learn, n.d.).

TF-IDF values are **weights, not probabilities**. They show the relative importance of words. They do not tell us the probability that a word will appear.

### 6. Difference Between fit, transform, and fit_transform

Scikit-learn uses these methods to learn and apply text representations.

| Method | Explanation |
|---|---|
| `fit()` | Learns the vocabulary and other information from the training documents. |
| `transform()` | Converts documents into numerical vectors using the learned information. |
| `fit_transform()` | Learns the vocabulary and converts the same documents in one step. |

**What are out-of-vocabulary words?**

An **out-of-vocabulary (OOV)** word is a word that was not found when the vectorizer learned its vocabulary.

For example, suppose the training vocabulary contains `car` but not `automobile`. When the vectorizer processes a new document containing `automobile`, it cannot represent that word using the existing vocabulary. A standard fixed-vocabulary vectorizer ignores it.

**What is data leakage?**

Data leakage happens when information from the test dataset is used during model training or preparation.

For example, fitting a vectorizer on both training and test documents allows information from the test set to influence the vocabulary and IDF values.

This can make evaluation results look better than they should. The test data is supposed to represent new documents that the model has not seen during training.

**Correct order of steps**

1. Split the dataset into training and test sets.
2. Fit the vectorizer using only the training documents.
3. Transform the training documents.
4. Transform the test documents using the same fitted vectorizer.
5. Train the model using the training data.
6. Evaluate the model using the test data.

Scikit-learn's `Pipeline` can help prevent data leakage by keeping vectorization and model training together (scikit-learn, n.d.).

### 7. Case Study: Swahili News Classification

Kastanos and Martin (2021) studied the classification of Swahili news articles. Their research compared different text representations and classification methods.

Swahili is an African language with fewer NLP resources than widely studied languages such as English. The study investigated how different methods perform when only a limited number of training documents have labels.

**Task and data**

The task was to classify Swahili news articles into six categories:

- Local news
- International news
- Finance
- Health
- Sports
- Entertainment

The researchers used a Swahili news classification dataset and compared several text representation methods.

**Methods compared**

| Method | Explanation |
|---|---|
| Word counts | Counts how often words appear. |
| TF-IDF | Gives higher weights to more specific words. |
| fastText | Uses numerical word representations. |
| Paragraph Vector | Learns numerical representations of complete documents. |
| Text GCN | Uses relationships between words and documents in a graph. |

**Results and discussion**

The study reported that word counts and TF-IDF provided useful results. However, the graph-based Text GCN approach achieved stronger results in the reported experiments.

The results also showed that fastText did not perform as well as the other methods in that particular experiment.

These findings suggest that the performance of a text representation depends on the dataset, language, and learning method. A more advanced method does not always perform better.

**What does the study show about BoW and TF-IDF?**

BoW and TF-IDF mainly focus on word occurrence and frequency. They do not directly understand the meaning of words.

For example, the words `car` and `automobile` have similar meanings, but a basic BoW or TF-IDF model treats them as separate features.

Two articles can also discuss the same event using different words. A word-count method may not recognize that they discuss the same subject.

The study shows why BoW and TF-IDF are useful starting methods but may have limitations when a task requires a deeper understanding of language meaning.

---

## References

1. David, D. (2020). *Swahili: News Classification Dataset* (Version 0.1). Zenodo. https://doi.org/10.5281/zenodo.4300294

2. Kastanos, A., & Martin, T. (2021). *Graph Convolutional Network for Swahili News Classification*. arXiv:2103.09325. https://arxiv.org/abs/2103.09325

3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

4. Nothman, J., Qin, H., & Yurchak, R. (2018). Stop word lists in free open-source software packages. *Proceedings of the Workshop for NLP Open Source Software*, 7–12. https://doi.org/10.18653/v1/W18-2502

5. SciPy. (n.d.). *Sparse matrices*. https://docs.scipy.org/doc/scipy/reference/sparse.html

6. scikit-learn. (n.d.). *Feature extraction: Text feature extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html

7. scikit-learn. (n.d.). *TfidfVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

8. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526