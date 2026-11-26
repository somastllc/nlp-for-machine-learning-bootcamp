# Part A: Research – Text Vectorization

## 1. What Is Text Vectorization?

Text vectorization is the process of converting text into numerical representations that machine learning algorithms can understand and process.

Computers cannot directly understand words as humans do. Therefore, text must be transformed into numbers before it can be used to train most machine learning models.

Common text vectorization techniques include:

- One-Hot Encoding
- Bag of Words (BoW)
- Term Frequency–Inverse Document Frequency (TF-IDF)
- N-gram representation

## 2. One-Hot Encoding

One-hot encoding represents each unique word using a binary vector. Each word receives a vector containing zeros and a single one at the position assigned to that word.

For example, if the vocabulary is:

`["cash", "free", "phone"]`

The vectors are:

| Word | One-Hot Vector |
|---|---|
| cash | [1, 0, 0] |
| free | [0, 1, 0] |
| phone | [0, 0, 1] |

**Advantages:**
- Simple to understand and implement.
- Gives each word a unique representation.

**Disadvantages:**
- Produces large and sparse vectors for large vocabularies.
- Does not capture relationships or similarities between words.

## 3. Bag of Words (BoW)

Bag of Words is a text representation technique that converts documents into numerical vectors based on the frequency of words.

It creates a vocabulary containing unique words from the documents and counts how many times each word appears in each document.

For example:

- Document 1: "I love Python"
- Document 2: "I love machine learning"

The vocabulary is:

`["I", "learning", "love", "machine", "Python"]`

The document-term matrix is:

| Document | I | learning | love | machine | Python |
|---|---:|---:|---:|---:|---:|
| Document 1 | 1 | 0 | 1 | 0 | 1 |
| Document 2 | 1 | 1 | 1 | 1 | 0 |

**Advantages:**
- Easy to implement.
- Works well as a basic text classification approach.
- Captures word frequency.

**Disadvantages:**
- Ignores word order and grammar.
- Does not understand context or meaning.
- Can produce large, sparse matrices.

## 4. Vocabulary and Sparsity

A vocabulary is the collection of unique words or tokens extracted from a set of documents.

The total number of words is not necessarily equal to the vocabulary size because repeated words are counted only once when calculating unique vocabulary.

A sparse matrix is a matrix containing many zero values. Text datasets often produce sparse matrices because each document contains only a small portion of the complete vocabulary.

Sparsity can be calculated as:

`Sparsity (%) = (Number of Zero Values / Total Matrix Cells) × 100`

Common ways to reduce vocabulary size include:

- `min_df`: Removes words appearing in fewer than a specified number of documents.
- `max_df`: Removes words appearing in more than a specified proportion or number of documents.
- `max_features`: Limits the vocabulary to the most frequent terms.

## 5. N-grams

N-grams are sequences of consecutive words used as text features.

There are three common types:

- **Unigrams:** Individual words, such as "machine".
- **Bigrams:** Two consecutive words, such as "machine learning".
- **Trigrams:** Three consecutive words, such as "natural language processing".

N-grams preserve some word-order information that ordinary Bag of Words ignores.

For example, the bigram "not good" provides more information about sentiment than treating "not" and "good" as completely independent words.

However, increasing the n-gram range can significantly increase vocabulary size and computational requirements.

## 6. TF-IDF

TF-IDF stands for Term Frequency–Inverse Document Frequency. It is a numerical weighting technique that measures how important a word is in a document compared with the entire collection of documents.

It gives higher importance to words that occur frequently in a particular document but are less common across the document collection.

### Origin of TF-IDF

TF-IDF developed from information retrieval and automatic document indexing research. Gerard Salton and Christopher Buckley published an influential paper, "Term-weighting Approaches in Automatic Text Retrieval", in 1988, discussing effective term-weighting methods for information retrieval.

### TF-IDF Formula

**Term Frequency (TF):**

`TF(t,d) = Count of term t in document d / Total terms in document d`

**Inverse Document Frequency (IDF):**

A common textbook formula is:

`IDF(t) = log(N / DF(t))`

Where:
- N = Total number of documents.
- DF(t) = Number of documents containing term t.

**TF-IDF:**

`TF-IDF(t,d) = TF(t,d) × IDF(t)`

A word receives a higher score when it is important to a particular document but uncommon in the overall collection.

### Scikit-learn TF-IDF

Scikit-learn's `TfidfVectorizer` uses a smoothed IDF formula by default:

`IDF(t) = log((1 + N) / (1 + DF(t))) + 1`

It also applies L2 normalization by default. Therefore, its TF-IDF values may differ from manually calculated textbook values.

### Why Use TF-IDF?

- Reduces the importance of very common words.
- Highlights informative words.
- Useful for search engines and document classification.
- Often performs well in traditional machine learning text classification tasks.

## 7. Fit, Transform, and Data Leakage

**Fit:** The vectorizer learns the vocabulary and, for TF-IDF, the IDF weights from the supplied training documents.

**Transform:** Converts new documents into vectors using the vocabulary and weights already learned during fitting.

**Fit_transform:** Learns the vocabulary and transforms the supplied documents in one operation.

### Correct Workflow

1. Split the dataset into training and testing sets.
2. Fit the vectorizer only on the training data.
3. Transform the training data.
4. Transform the testing data using the same fitted vectorizer.
5. Train and evaluate the model.

### What Is Data Leakage?

Data leakage happens when information from the test dataset is used during model training or preprocessing.

For example, fitting a vectorizer on both training and testing documents allows the vectorizer to learn vocabulary information from the test set.

This can make the evaluation less reliable.

To prevent data leakage, fit the vectorizer only on training data and use it to transform the test data.

## 8. Case Study: Text Vectorization for the Somali Language

Somali is an example of a low-resource language in Natural Language Processing (NLP). Compared with languages such as English, Somali has fewer publicly available annotated datasets and language-processing resources.

A 2023 research paper by Badel and colleagues introduced a Somali Information Retrieval Corpus containing 2,335 documents. The researchers highlighted the shortage of dedicated Somali-language resources and the importance of building datasets for future NLP applications.

A 2026 study by Ali explored Somali news classification using TF-IDF and a Linear Support Vector Machine (SVM). The research used a curated dataset of 7,240 Somali news articles covering eight categories and reported a 94.20% accuracy for its best configuration.

These studies demonstrate how traditional text vectorization techniques can support NLP applications in Somali.

### Possible Applications

- Somali news classification.
- Spam message detection.
- Sentiment analysis of Somali social media posts.
- Somali information retrieval and search engines.
- Categorization of Somali documents.

### Challenges

- Limited annotated Somali datasets.
- Differences in spelling and word forms.
- Limited language-specific preprocessing tools.
- Dialect and vocabulary variations.

For Somali text classification, TF-IDF with word unigrams and bigrams can provide a useful baseline. However, preprocessing should be designed carefully to preserve important words and expressions.

## 9. Conclusion

Text vectorization is an essential step in Natural Language Processing because it converts human language into numerical features that machine learning algorithms can process.

One-Hot Encoding and Bag of Words provide simple representations, while N-grams preserve some word-order information. TF-IDF improves word weighting by considering how frequently terms appear across documents.

For practical text classification tasks, TF-IDF combined with unigram and bigram features is a useful starting point. Proper training and testing separation is also essential to prevent data leakage.

## References

1. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

2. Scikit-learn. *CountVectorizer Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

3. Scikit-learn. *TfidfVectorizer Documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

4. Badel, A., Zhong, T., Tai, W., & Zhou, F. (2023). *Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources*. Proceedings of EMNLP 2023. https://aclanthology.org/2023.emnlp-main.462/

5. Ali, M. M. (2026). *Resource-efficient Somali news classification using TF-IDF and linear SVM: a supervised baseline on a curated news corpus*. Scientific Reports. https://doi.org/10.1038/s41598-026-68159-5