A1. Text Vectorization and One-Hot Encoding:
    Text vectorization is the process of converting text into numerical features that machine learning models can understand and process. Most ML models require documents to have the same number of features because they use fixed-size numerical inputs.

    One-hot encoding represents each word as a binary vector containing zeros and one, where the position of one identifies the word. However, it is not enough to represent a whole document because it represents individual words separately and does not capture their meaning or relationships.

    For example, a spam detection system converts emails into numerical features so that a machine learning model can classify them as spam or legitimate messages.

A2. Bag of Words (BoW) is a text representation method that converts documents into numerical vectors based on 
    word frequency,  without   considering word order.

    A document-term matrix represents documents in rows, unique words in columns, and the frequency of each word in the corresponding cells. It is called a "bag" because it ignores the order and structure of words.

    For example, losing word order causes little harm in spam detection, where word frequency can help identify spam messages. However, it can cause problems in sentiment analysis because "good" and "not good" have different meanings, but BoW may not capture this difference.

A3. Vocabulary Size and Sparsity
    Vocabulary size is the total number of unique words in a collection of documents. Sparsity occurs when most cells in a document-term matrix contain zero because each document uses only a small portion of the entire vocabulary.

    Real document-term matrices become large because they may contain thousands of unique words across many documents. Sparse matrices save memory by storing only non-zero values and their positions instead of storing every zero.

    Settings such as min_df remove words appearing in too few documents, max_df removes words appearing in too many documents, and max_features limits the vocabulary size. Preprocessing also changes the vocabulary: lowercasing combines uppercase and lowercase forms, stopword removal eliminates common words, and stemming combines related word forms into shared stems

A4. N-grams are groups of consecutive words used to represent text while preserving some word order. A unigram contains one word (e.g., "good"), a bigram contains two consecutive words (e.g., "not good"), and a trigram contains three consecutive words (e.g., "not very good").

N-grams help capture context that Bag of Words may miss. For example, "good" and "not good" have different meanings, and a bigram can preserve the negation. However, larger n-grams increase vocabulary size, memory usage, and computational cost, and may create sparse matrices because many word combinations occur rarely.

A5. TF-IDF (Term Frequency–Inverse Document Frequency) is a text-weighting method that measures how important a word is in a document compared to the entire collection. The idea of IDF was introduced by Karen Spärck Jones in her 1972 research on term specificity in information retrieval.

The textbook formula is:

TF-IDF(t, d) = TF(t, d) × IDF(t)

IDF(t) = log(N / df(t))

Here, N is the total number of documents and df(t) is the number of documents containing the term. Rare words receive higher IDF weights because they are more useful for distinguishing documents than very common words.

Scikit-learn's TfidfVectorizer uses a smoothed IDF formula by default: log((1 + N) / (1 + df(t))) + 1. It also applies L2 normalization by default, making each document vector have a unit length. TF-IDF values are weights, not probabilities, because they measure relative term importance and do not represent the likelihood of an event.

A6. fit() learns the vocabulary and statistical information from the training documents. transform() converts new documents into numerical vectors using the learned vocabulary, while fit_transform() performs both operations together. Out-of-vocabulary (OOV) words are ignored because they are not part of the learned vocabulary.

Data leakage occurs when information from the test set is used during training or vectorizer fitting. This can make evaluation results appear better than the model's actual performance because the model indirectly gains information from data it should not have seen.

The correct process is to split the data first, fit the vectorizer only on the training data, transform the training data, train the model, and then transform the test data using the same fitted vectorizer before evaluation.

A6. Case Study – Amharic Text Representation
Endalie, Haile, and Taye (2023) conducted a study titled Deep Learning-Based Idiomatic Expression Recognition for the Amharic Language. The task was to identify whether Amharic phrases were idiomatic or literal.

The researchers collected 3,300 phrases from two Amharic books: 1,700 idiomatic and 1,600 non-idiomatic expressions. They compared One-Hot Encoding, TF-IDF, and FastText embeddings combined with a CNN.

The results showed that CNN with FastText achieved 80% accuracy, compared with 74% for TF-IDF and 71.3% for One-Hot Encoding. FastText performed better because it represents semantic relationships between words and phrases.

This study demonstrates a limitation of traditional representations such as TF-IDF: they mainly measure word frequency and do not capture deeper meanings. For example, an idiomatic expression may have a meaning completely different from the literal meanings of its individual words.

Reference:
Endalie, D., Haile, G., & Taye, W. (2023). Deep learning-based idiomatic expression recognition for the Amharic language. 
PLOS ONE, 18(12). https://doi.org/10.1371/journal.pone.0295339

# Part B: Practical Report – Text Vectorization

## Task 1: One-Hot Encoding

One-hot encoding represents each word using a binary vector containing one `1` and zeros in all other positions. Using the given vocabulary, each word has a vector of length six. The sentence "win free cash" produces three vectors because it contains three words, and each word is represented separately.

## Task 2: Bag of Words and Document-Term Matrix

Bag of Words represents documents using word frequencies without preserving word order. The Document-Term Matrix contains documents as rows, vocabulary words as columns, and word counts as cell values.

### Document-Term Matrix

The vocabulary contains 32 words, and the matrix has a shape of (5, 32).

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

The word "win" has a count of 2 in D1 because it appears twice with different capitalization: "Win" and "WIN". CountVectorizer converts text to lowercase by default. The vectors for "win free phone" and "phone free win" are identical because Bag of Words ignores word order.

## Task 4: Preprocessing Inside the Vectorizer

The default CountVectorizer converts words to lowercase and removes punctuation. It also ignores single-character words such as "a". With `lowercase=False`, uppercase and lowercase forms such as "Win" and "WIN" are treated as different features. The English stopword list removes common words, including "a", "to", and "now"; when applied to all documents, it also removes "not", which changes the meaning of D5 by removing its negative expression.

## Task 5: Limiting the Vocabulary

The full vocabulary contains 32 words. Using `min_df=2` reduces it to 7 words: free, lecture, notes, now, the, today, and win. Using `max_features=5` keeps five frequent words: free, lecture, now, the, and win. These settings reduce the vocabulary size but may remove useful information from less frequent words.

## Task 6: N-grams

Unigrams represent individual words, while bigrams represent pairs of consecutive words. In D5, the bigram "not good" preserves an important part of the reviewer's negative opinion. Across all documents, the vocabulary grows from 32 unigrams to 67 unigrams and bigrams, and 98 unigrams, bigrams, and trigrams. Larger n-grams increase the vocabulary size and computational cost.

## Task 7: TF-IDF

TF-IDF gives higher weights to words that are important in a document but less common across the document collection. In D2, every word appears once, but some words receive higher weights because they are less frequent across the other documents. In D1, "win" receives the highest weight because it appears twice and is relatively uncommon in the collection.

### D2: Count vs TF-IDF

| Word | Count | TF-IDF Weight |
|---|---:|---:|
| call | 1 | 0.389 |
| cash | 1 | 0.389 |
| entry | 1 | 0.389 |
| free | 1 | 0.314 |
| now | 1 | 0.314 |
| prizes | 1 | 0.389 |
| today | 1 | 0.314 |
| win | 1 | 0.314 |

## Task 8: TF-IDF by Hand

For D2, the total number of words is 8, and the total number of documents is 5.

The formulas are:

TF = Word Count / Total Words in D2

IDF = log(N / DF)

TF-IDF = TF × IDF

### Word: cash

- Count = 1
- TF = 1 / 8 = 0.125
- DF = 1
- IDF = log(5 / 1) = 1.6094
- TF-IDF = 0.125 × 1.6094 = 0.2012

### Word: today

- Count = 1
- TF = 1 / 8 = 0.125
- DF = 2
- IDF = log(5 / 2) = 0.9163
- TF-IDF = 0.125 × 0.9163 = 0.1145

The hand calculations rank "cash" higher than "today", which agrees with the ranking from scikit-learn. However, the exact values differ because scikit-learn applies IDF smoothing and L2 normalization by default.

## Task 10: Fit and Transform

The vectorizer was fitted using D1–D4 only and then used to transform N1 and N2. Words that were not present in the training vocabulary were treated as out-of-vocabulary (OOV) words. The OOV words were "your" and "prize" in N1, and "is", "the", "moved", and "tomorrow" in N2. The word "prize" is unknown because the training vocabulary contains "prizes" as a different word. Stemming or lemmatization can help map related word forms to a common representation.

## Task 11: Data Leakage

Data leakage occurs when information from the test set is used during the training or preparation process. Fitting a vectorizer on both training and test documents adds test-only words to the vocabulary. In this experiment, the leaked words were "is", "moved", "prize", "the", "tomorrow", and "your". The correct approach is to fit the vectorizer on training data only and use `transform()` on test data.

## Task 12: Bag of Words Cannot Understand Meaning

The CountVectorizer treats "tasty", "delicious", and "terrible" as separate features. The cosine similarity between every pair of reviews is 0.75, even though "tasty" and "delicious" have similar meanings while "terrible" has a different meaning. This demonstrates that Bag of Words measures word overlap rather than semantic meaning. Word embeddings such as Word2Vec and FastText can capture semantic relationships between words more effectively.

## Spam Detector Recommendation

For a spam detector, I would start with TF-IDF because it gives higher weights to informative words that are less common across documents. I would also test unigrams and bigrams because phrases such as "free prize" and "claim now" may provide useful spam signals. I would avoid removing stopwords blindly because words such as "not" can change the meaning of a message. I would use vocabulary limits carefully because removing rare words may remove useful spam-related terms. Finally, I would compare different configurations using validation data and select the one with the best evaluation performance.
