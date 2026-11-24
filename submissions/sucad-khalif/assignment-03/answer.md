# Part A – Research

## 1. What is Text Vectorization?

Text vectorization is the process of converting written text into numerical representations that Machine Learning algorithms can understand and process.
Most Machine Learning models work with numerical data rather than raw text. 

One basic technique is **one-hot encoding**. It assigns a unique position to every word in a vocabulary. Each word is represented by a vector containing zeros, except for one position containing a value of one.

For example, consider a customer support system that processes messages using the vocabulary `[delivery, payment, refund]`.

| Word | Delivery | Payment | Refund |
|---|---:|---:|---:|
| delivery | 1 | 0 | 0 |
| payment | 0 | 1 | 0 |
| refund | 0 | 0 | 1 |

One-hot encoding represents individual words rather than complete documents. A sentence containing several words produces several separate vectors. It also creates large vectors when the vocabulary contains many words. Therefore, other methods, such as Bag of Words, are used to represent complete documents.

## 2. Bag of Words and the Document-Term Matrix

Bag of Words  is a text representation technique that converts documents into numerical vectors by counting the occurrences of words in each document.

A document-term matrix is a table that organizes these numerical representations. It contains the following elements:

- **Rows:** Represent individual documents.
- **Columns:** Represent unique words in the vocabulary.
- **Cells:** Contain the number of times each word appears in a document.

For example, consider two customer messages:

- Document 1: "I need a refund."
- Document 2: "I need payment support."

Their Bag of Words representation could be:

| Document | I | need | a | refund | payment | support |
|---|---:|---:|---:|---:|---:|---:|
| Document 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| Document 2 | 1 | 1 | 0 | 0 | 1 | 1 |

It is called a "bag" because the method considers word frequency without preserving the original order of words.


## 3. Vocabulary Size and Sparsity

Vocabulary size refers to the total number of unique terms represented in a document collection. Each vocabulary term corresponds to a feature or column in the document-term matrix.


Sparse matrices are stored efficiently by recording non-zero values together with their positions rather than storing every zero. This reduces memory usage and improves computational efficiency. Scikit-learn uses sparse matrix representations for text vectorization.

Several settings can control vocabulary size:

| Setting | Description |
|---|---|
| `min_df` | Removes terms that appear in fewer than a specified number or proportion of documents. |
| `max_df` | Removes terms that appear in more than a specified number or proportion of documents. |
| `max_features` | Limits the vocabulary to the most frequent specified number of terms. |

Preprocessing also changes the vocabulary:

- **Lowercasing:** Converts uppercase and lowercase versions of a word into one form, reducing duplicate entries.
- **Stopword removal:** Removes selected common words that may contribute little to a particular task.
- **Stemming:** Reduces related word forms to a common stem, helping decrease the number of separate features.


## 4. What are N-grams?

N-grams are sequences of neighboring words used to represent text while preserving some information about word order.

The main types include:

| Type | Description | Example |
|---|---|---|
| Unigram | One word at a time. | `machine` |
| Bigram | Two consecutive words. | `machine learning` |
| Trigram | Three consecutive words. | `machine learning model` |

N-grams help identify expressions whose meanings depend on neighboring words.

For example, the bigram "not good" expresses a negative opinion, while the unigram "good" usually expresses a positive opinion. Including bigrams helps a Machine Learning model recognize this difference.



## 5. What is TF-IDF?

TF-IDF stands for Term Frequency–Inverse Document Frequency. It is a numerical weighting technique used to estimate how important a word is to a particular document compared with an entire collection of documents.

The idea of inverse document frequency is associated with Karen Spärck Jones (1972). Her research on statistical term specificity showed that words appearing in fewer documents can be more useful for distinguishing relevant documents from a large collection.

TF-IDF combines two components:

- **Term Frequency (TF):** Measures how frequently a term appears in a particular document.
- **Inverse Document Frequency (IDF):** Measures how rare a term is across the complete document collection.

The textbook formulas are:

\[
TF(t,d)=\frac{\text{Count of term }t\text{ in }d}
{\text{Total number of terms in }d}
\]

\[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
\]

\[
TF\text{-}IDF(t,d)=TF(t,d)\times IDF(t)
\]

Where:

- \(t\) represents a term.
- \(d\) represents a document.
- \(N\) represents the total number of documents.
- \(df(t)\) represents the number of documents containing the term.

Rare words receive higher IDF values because they appear in fewer documents and can help distinguish one document from another. Common words receive lower IDF values because they provide less information about the differences between documents.

### TF-IDF in Scikit-learn

Scikit-learn's `TfidfVectorizer` uses a slightly different implementation from the basic textbook formula.

Its default IDF formula is:

\[
IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

The main differences include:

1. **Smoothing:** It adds one to the numerator and denominator to avoid zero divisions and extreme values.
2. **Normalization:** By default, it applies L2 normalization so that each non-zero document vector has a Euclidean length of one.
3. **Term frequency:** It uses raw term counts by default, although logarithmic term-frequency scaling can be enabled.

TF-IDF values are weights, not probabilities. They measure the relative importance of terms within documents and are not restricted to representing the likelihood of an event. For example, a weight of 0.70 does not mean that a word has a 70% probability of appearing.

## 6. Fit, Transform and Data Leakage

In text vectorization, `fit`, `transform` and `fit_transform` perform different operations.

| Method | Description |
|---|---|
| `fit()` | Learns the vocabulary and other required information from the supplied training documents. |
| `transform()` | Converts documents into numerical vectors using an already learned vocabulary. |
| `fit_transform()` | Learns the vocabulary and transforms the supplied documents in one operation. |



**Data leakage** occurs when information from the test dataset is used during the training process. In vectorization, this can happen when a vocabulary is learned from both training and test documents before model evaluation.

This creates an unrealistic evaluation because the model's preprocessing has already received information from the test data. As a result, performance measurements may appear better than they would be on genuinely unseen documents.

The correct order is:

1. Split the dataset into training and testing sets.
2. Fit the vectorizer using only the training documents.
3. Transform the training documents using the fitted vectorizer.
4. Transform the test documents using the same fitted vectorizer.
5. Train and evaluate the Machine Learning model.

This procedure prevents information from the test set from influencing the learned vocabulary and feature extraction process.

## 7. Case Study: Amharic Idiomatic Expression Recognition

Endalie, Haile and Taye (2023) conducted a research study titled *Deep Learning-Based Idiomatic Expression Recognition for the Amharic Language*. The research focused on identifying idiomatic and non-idiomatic expressions in Amharic, an Ethiopian language.

### Research Task and Dataset

The main task was to determine whether an Amharic phrase was used as an idiomatic expression or in its literal meaning.

The researchers collected 3,300 expressions from two Amharic books. The dataset contained:

- 1,700 idiomatic expressions.
- 1,600 non-idiomatic expressions.

The researchers divided the data into training, validation and testing sets using an 80%, 10% and 10% split.

### Text Representations Compared

The study investigated different numerical representations, including:

- One-hot encoding.
- TF-IDF.
- FastText word embeddings.

The proposed approach combined FastText word embeddings with a Convolutional Neural Network (CNN). The researchers also compared the proposed model with traditional machine learning classifiers.

### Results

The comparison of different text representations reported the following accuracy results:

| Text representation | Accuracy |
|---|---:|
| One-hot encoding | 71.3% |
| TF-IDF | 74% |
| FastText with CNN | 80% |

The FastText-based CNN achieved the highest accuracy among the compared representations.

### Discussion

This study demonstrates an important limitation of Bag of Words and TF-IDF methods. These approaches mainly represent words through their occurrence and statistical importance rather than their actual meanings.

Idiomatic expressions are particularly difficult because the meaning of a complete phrase may differ from the literal meanings of its individual words. FastText provides word representations based on character-level information and learned linguistic patterns, helping the system capture relationships that basic frequency-based representations may miss.

Therefore, the research supports the importance of using richer word representations when an NLP task requires understanding expressions and their meanings.

**Overall, the study shows that numerical word counts and weights can be useful for text classification, but they may not be sufficient for tasks that depend strongly on semantic meaning.**

## References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/

2. Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

4. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/EB026526

5. Scikit-learn Developers. (2026). *CountVectorizer*. Scikit-learn Documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

6. Scikit-learn Developers. (2026). *TfidfVectorizer*. Scikit-learn Documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

7. Scikit-learn Developers. (2026). *Common pitfalls and recommended practices: Data leakage*. https://scikit-learn.org/stable/common_pitfalls.html

8. Endalie, D., Haile, G., & Taye, W. (2023). Deep learning-based idiomatic expression recognition for the Amharic language. *PLOS ONE, 18*(12), e0295339. https://doi.org/10.1371/journal.pone.0295339