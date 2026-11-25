### 1. What is Vectorization?

Vectorization is the process of converting text into numerical vectors so that machine learning algorithms can process and analyze text data.

All text documents must be represented using the same feature dimensions because machine learning algorithms require consistent input sizes. Using the same vocabulary ensures that each position in a vector represents the same feature across all documents.

**One-hot encoding** is a technique that represents each word as a binary vector. One position contains `1`, while all other positions contain `0`.

For example, with the vocabulary `[cat, dog, fish]`:
- cat = `[1, 0, 0]`
- dog = `[0, 1, 0]`
- fish = `[0, 0, 1]`

One-hot encoding is not enough to represent a whole document efficiently because it creates a separate vector for each word. It also does not capture semantic relationships between words.

**Real-life example:** In spam detection, emails are converted into numerical vectors before a machine learning model analyzes them and predicts whether they are spam or not spam.

### 2. Bag of Words and Document-Term Matrix

Bag of Words (BoW) is a text representation technique that converts documents into numerical vectors by counting how many times each vocabulary word appears in a document.

A Document-Term Matrix (DTM) is a table that represents the frequency of vocabulary words across multiple documents.

- **Rows** represent documents.
- **Columns** represent vocabulary terms.
- **Cells** contain the frequency or count of each term in a document.

It is called a "Bag of Words" because it focuses on word frequencies rather than word order.

Losing word order may be harmless in spam detection because individual words such as "FREE" and "WIN" can help identify spam messages. However, it can be harmful in sentiment analysis. For example, "The movie is good" and "The movie is not good" have different meanings, but a basic Bag of Words representation may fail to capture the importance of word order and negation.

### 3. Vocabulary Size and Sparsity

Vocabulary size is the total number of unique words in a collection of documents. A word that appears in multiple documents is counted only once in the vocabulary.

Sparsity means that a matrix contains many zero values compared to its total number of cells. A Document-Term Matrix can become large and sparse because the vocabulary grows as more documents are added, but each document usually contains only a small portion of the vocabulary.

Sparse matrix storage is efficient because it stores non-zero values and their positions instead of storing every zero, reducing memory usage.

The following settings can reduce vocabulary size:

- `min_df`: Removes words that appear in fewer than a specified number of documents.
- `max_df`: Removes words that appear in more than a specified proportion or number of documents.
- `max_features`: Limits the vocabulary to a maximum number of features.

Preprocessing also affects vocabulary size. Lowercasing combines words with different capitalization, stopword removal removes common words, and stemming reduces related word forms to a common stem.
### 4. N-grams

N-grams are groups of consecutive words used to represent text while preserving some word order.

- **Unigrams:** Individual words, such as "I", "love", and "NLP".
- **Bigrams:** Groups of two consecutive words, such as "I love" and "love NLP".
- **Trigrams:** Groups of three consecutive words, such as "I love NLP".

N-grams preserve partial word order because they capture neighboring words. For example, the bigrams "not good" and "good not" are different, even though they contain the same individual words.

Larger n-grams can increase vocabulary size, memory usage, and computational cost because they create more possible word combinations.
### 5. TF-IDF

TF-IDF stands for Term Frequency–Inverse Document Frequency. It is a numerical weighting technique used to measure how important a word is in a document compared with a collection of documents.

**Term Frequency (TF)** measures how often a term appears in a document. A basic formula is:

\[
TF(t,d)=\frac{\text{Count of term }t\text{ in document }d}{\text{Total number of words in document }d}
\]

**Inverse Document Frequency (IDF)** measures how rare or common a term is across the document collection. The basic formula is:

\[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
\]

Here, \(N\) is the total number of documents, and \(df(t)\) is the number of documents containing the term.

The basic TF-IDF formula is:

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

Rare words generally receive higher IDF values, while very common words receive lower IDF values.

The IDF concept was developed in Information Retrieval research, with Karen Spärck Jones making an important contribution in 1972.

Scikit-learn's `TfidfVectorizer` uses a smoothed IDF formula by default:

\[
IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

It also applies L2 normalization by default. TF-IDF values are weights, not probabilities, because they measure term importance rather than the probability of an event.

### 6. Fit, Transform, and Data Leakage

- **fit():** Learns the vocabulary and other required information from the training data.
- **transform():** Converts documents into numerical vectors using the vocabulary learned during fitting.
- **fit_transform():** Performs both fitting and transforming in one step.
- **Out-of-vocabulary (OOV) words:** Words that were not present in the training vocabulary. CountVectorizer ignores these words during transformation.
- **Data leakage:** Occurs when information from the test data is used during training or preprocessing, leading to unreliable evaluation.

The correct approach is to fit the vectorizer on the training data only, then transform both the training and test data using the learned vocabulary.
### 7. Case Study: Somali News Classification

The research paper studied Somali news classification using machine learning.

- **Task:** Classifying Somali news articles into eight categories, including Politics, Health, Sports, Business, and Technology.
- **Dataset:** 7,240 Somali news articles.
- **Text representation:** Word-level TF-IDF using unigrams.
- **Machine learning model:** Linear Support Vector Machine (Linear SVM).
- **Result:** The model achieved 94.20% accuracy.

**Connection to semantic limitations:**

TF-IDF represents text using word frequencies and importance weights. However, it does not fully understand the meaning or context of words. For example, two Somali sentences can express similar meanings but use different words. TF-IDF may represent them as different vectors. Contextual embeddings such as BERT can capture more semantic information.


# Practical Part B – Vectorization

## Task 1: One-Hot Encoding

One-hot encoding represents each word as a binary vector. Each word has a unique position in the vocabulary. The vector contains 1 at the word's position and 0 elsewhere.

## Task 2: Bag of Words and Document-Term Matrix

Bag of Words represents documents using word frequencies. It creates a Document-Term Matrix (DTM), where rows represent documents and columns represent vocabulary terms.

The vocabulary size was 32, and the DTM shape was (5, 32).

The DTM was:

```text
D1: [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,2]
D2: [0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1]
D3: [1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0]
D4: [0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0]
D5: [0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,2,0,0,1,0,1,0]
```

Bag of Words ignores word order and semantic meaning.

## Task 3: Vocabulary and Document Frequency

The total number of tokens was 42, and the vocabulary size was 32.

Document Frequency (DF) measures the number of documents containing a specific word, regardless of how many times it appears in each document.

The words appearing in more than one document were:

- free: 2
- lecture: 2
- notes: 2
- now: 2
- the: 2
- today: 2
- win: 2

## Task 4: CountVectorizer Settings

The `lowercase=False` setting makes CountVectorizer case-sensitive. Therefore, words such as "WIN" and "Win" are treated as different features.

The `stop_words="english"` setting removes common English stop words. However, removing words such as "not" can change the meaning of a sentence and negatively affect sentiment analysis.

## Task 5: min_df and max_features

With `min_df=2`, the vocabulary size decreased from 32 to 7 because only words appearing in at least two documents were retained.

With `max_features=5`, CountVectorizer selected only five features.

A word appearing many times in one document can still be removed by `min_df=2` if it does not appear in at least two documents.

## Task 6: N-grams

Unigrams represent individual words, while bigrams represent pairs of consecutive words.

For example, "not good" is a bigram that can help preserve an important meaning in sentiment analysis.

Increasing the n-gram range can increase the number of features and make the matrix more sparse.

## Task 7: TF-IDF

TF-IDF measures the importance of a word in a document relative to the entire document collection.

The following table shows the Count and TF-IDF values for Document 2:

| Word | Count | TF-IDF |
|---|---:|---:|
| call | 1 | 0.39 |
| cash | 1 | 0.39 |
| entry | 1 | 0.39 |
| free | 1 | 0.31 |
| now | 1 | 0.31 |
| prizes | 1 | 0.39 |
| today | 1 | 0.31 |
| win | 1 | 0.31 |

Rare words receive higher IDF values because they provide more distinguishing information across documents.

## Task 8: Manual TF-IDF Calculation

For Document 2:

- Number of documents (N) = 5
- Count of cash = 1
- DF of cash = 1
- Count of today = 1
- DF of today = 2

Scikit-learn uses the smoothed IDF formula:

`IDF = log((1 + N) / (1 + DF)) + 1`

For cash:

`IDF = log(6 / 2) + 1 = 2.0986`

For today:

`IDF = log(6 / 3) + 1 = 1.6931`

The normalized TF-IDF values were approximately:

- cash: 0.3891
- today: 0.3140

Normalization makes document vectors comparable in scale and supports similarity calculations.

## Task 9: Sparse Matrix Analysis

The DTM contained:

- Total cells: 160
- Non-zero cells: 39
- Zero cells: 121
- Sparsity: 75.62%

A sparse matrix contains many zero values. Sparse matrix storage is useful because it saves memory by storing only non-zero values.

## Task 10: Out-of-Vocabulary (OOV) Words

Out-of-vocabulary words are words that do not exist in the vocabulary learned during training.

In the experiment, "prize" was not recognized because the training vocabulary contained "prizes" instead.

Stemming can help map related word forms to a common root, reducing vocabulary variation. However, stemming does not guarantee that every OOV word will be recognized.

## Task 11: Data Leakage

The vocabulary size was:

- Training documents only: 26
- All documents: 32

Fitting a vectorizer on both training and test data introduces data leakage because information from the test set influences feature learning.

The correct approach is to fit the vectorizer on training data only and transform the test data using the same fitted vectorizer.

## Task 12: Cosine Similarity

The cosine similarity matrix was:

```text
[[1.00 0.75 0.75]
 [0.75 1.00 0.75]
 [0.75 0.75 1.00]]
```

All three reviews shared the words "the", "food", and "is".

CountVectorizer does not understand semantic meaning or synonyms. Therefore, it treats "tasty", "delicious", and "terrible" as different features, even though "tasty" and "delicious" have similar meanings.

Cosine similarity measures similarity based on vector direction, not actual human understanding of language.

---

## Conclusion

These practical exercises demonstrated how text can be converted into numerical representations using One-Hot Encoding, Bag of Words, CountVectorizer, and TF-IDF.

They also demonstrated important NLP concepts such as n-grams, sparse matrices, OOV words, data leakage, and cosine similarity.

### Task 1: One-Hot Encoding

One-hot encoding represents each word as a binary vector.
Each word has a unique position in the vocabulary.
The vector contains 1 at the word's position and 0 elsewhere.

### Task 2: Bag of Words

Bag of Words represents documents using word frequencies.
It creates a Document-Term Matrix (DTM).
However, it ignores word order and semantic meaning.

### Task 3: Vocabulary and Document Frequency

The total number of tokens was 42, and the vocabulary size was 32.

Document Frequency (DF) measures the number of documents containing a specific word, regardless of how many times it appears in each document.