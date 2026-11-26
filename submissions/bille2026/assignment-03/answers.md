ASSIGNMENT 03 – VECTORIZATION

Course: Natural Language Processing (NLP)
Assignment: 03 – Vectorization
Part: A – Research

⸻

1. Introduction to Text Vectorization

Text vectorization is the process of converting text into numerical representations that Machine Learning (ML) algorithms can understand and process.

Computers can store and display text, but most traditional Machine Learning algorithms cannot directly use raw text as input. They require numerical features.

Vectorization converts documents into numerical vectors. Each document is represented using the same set of features, which allows ML models to compare and analyze them.

For example, consider the vocabulary:

[free, meeting, phone, win]

A simple one-hot representation of the word “free” is:

[1, 0, 0, 0]

The value 1 indicates that the word is present, while 0 indicates that it is not.

One-hot encoding is useful for representing individual words, but it does not conveniently represent the complete meaning or structure of a document. It also creates separate features for every word and does not capture relationships between words.

Therefore, other techniques such as Bag of Words and TF-IDF are commonly used to represent complete documents.

2. One-Hot Encoding, Bag of Words, and Document-Term Matrix

2.1 One-Hot Encoding

One-hot encoding represents each word using a vector containing zeros and one one.

For example, using the vocabulary:

[good, bad, excellent, poor]

The word “good” can be represented as:

[1, 0, 0, 0]

The word “bad” can be represented as:

[0, 1, 0, 0]

One-hot encoding is simple, but it has limitations:

* It creates a separate vector for each word.
* It does not represent the complete document efficiently.
* It does not capture semantic relationships between words.
* It can create large and sparse representations when the vocabulary is large.

2.2 Bag of Words (BoW)

Bag of Words is a text vectorization technique that represents a document by counting how many times each vocabulary word appears.

It is called a “bag” because it ignores word order and focuses on word occurrence and frequency.

For example:

* Document 1: “I like machine learning”
* Document 2: “I like machine learning learning”

Using the vocabulary:

[I, like, machine, learning]

The vectors are:

* Document 1: [1, 1, 1, 1]
* Document 2: [1, 1, 1, 2]

The second document has a value of 2 for “learning” because that word appears twice.

2.3 Document-Term Matrix (DTM)

A Document-Term Matrix is a table that represents documents as rows and vocabulary terms as columns.

Each cell contains the number of times a particular word appears in a document.

Example:

Document	machine	learning	data
D1	1	1	0
D2	1	2	1
D3	0	1	2

In this matrix:

* Rows represent documents.
* Columns represent vocabulary words.
* Values represent word frequencies.

BoW and DTM are useful for tasks such as spam detection and document classification.

However, they ignore word order. For example, “good movie” and “movie good” receive the same word-count representation.

This can cause problems in sentiment analysis because “good” and “not good” have different meanings, even though they contain the same important word.

3. Vocabulary Size and Sparsity

3.1 Vocabulary Size

Vocabulary refers to the collection of unique words identified from a group of documents.

Vocabulary size is the total number of unique words in that collection after the selected preprocessing and filtering steps.

For example:

* Document 1: “I like Python”
* Document 2: “I like Machine Learning”

The unique vocabulary is:

[I, like, Python, Machine, Learning]

The vocabulary size is 5.

A larger vocabulary creates more features in the document-term matrix. This can increase memory usage and computational cost.

3.2 Sparsity

A document-term matrix is often sparse because most documents contain only a small number of words from the complete vocabulary.

For example, a vocabulary may contain 10,000 words, but one document may contain only 100 of those words. Most matrix values will therefore be zero.

Sparse matrices are useful because they store non-zero values more efficiently than storing every zero.

Scikit-learn’s text vectorizers use sparse representations, commonly CSR (Compressed Sparse Row) format, to reduce memory usage.

3.3 Vocabulary Control

Scikit-learn provides parameters that help control vocabulary size:

* min_df: Removes words that appear in fewer than the specified number or proportion of documents.
* max_df: Removes words that appear in more than the specified number or proportion of documents.
* max_features: Limits the vocabulary to a maximum number of features.

These parameters can reduce unnecessary features and make processing more efficient.

3.4 The Role of Preprocessing

Preprocessing affects vocabulary size and the quality of text features.

Common preprocessing techniques include:

* Lowercasing: Converts all text to lowercase.
* Stop-word removal: Removes selected common words that may provide little useful information for a particular task.
* Stemming: Reduces words to their root or stem form.
* Lemmatization: Converts words into their dictionary base form.

Preprocessing must be selected carefully. Removing a word that carries important meaning can reduce model performance. For example, removing “not” may change the meaning of a negative sentence.

4. N-grams

An n-gram is a sequence of consecutive words or tokens in a text.

N-grams help preserve some word-order information that ordinary Bag of Words ignores.

There are three common types:

Unigram: A single word.

Example: “I love machine learning”

Unigrams:

I, love, machine, learning

Bigram: Two consecutive words.

Bigrams:

I love, love machine, machine learning

Trigram: Three consecutive words.

Trigrams:

I love machine, love machine learning

Why Are N-grams Important?

N-grams can capture short phrases and relationships between neighboring words.

For example:

* “good”
* “not good”

A unigram representation may include the word “good” in both sentences. A bigram representation can capture the phrase “not good” as a separate feature.

This can be useful in sentiment analysis and text classification.

Limitations of N-grams

Using larger n-grams can create a much larger vocabulary because more word combinations become possible.

This can increase:

* Memory usage.
* Computational cost.
* The number of rare features.
* The risk of overfitting when training data is limited.

Scikit-learn’s ngram_range parameter controls which n-gram sizes are included.

For example:

ngram_range=(1, 2)

This includes both unigrams and bigrams.

5. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF is a numerical technique used to measure how important a word is within a document compared with a collection of documents.

Unlike basic Bag of Words, which counts word occurrences, TF-IDF reduces the weight of words that appear in many documents and gives more importance to words that are more distinctive.

5.1 Term Frequency (TF)

Term Frequency measures how frequently a word appears in a particular document.

A simple formula is:

TF(t, d) = Number of times term t appears in document d / Total number of terms in document d

A word that appears many times in a document receives a higher term frequency.

5.2 Inverse Document Frequency (IDF)

Inverse Document Frequency measures how rare or common a word is across the document collection.

The traditional textbook formula is:

IDF(t) = log(N / df(t))

Where:

* N = Total number of documents.
* df(t) = Number of documents containing term t.

If a word appears in almost every document, its IDF value becomes low because it is less useful for distinguishing documents.

If a word appears in only a few documents, its IDF value becomes higher.

5.3 TF-IDF Formula

The basic TF-IDF formula is:

TF-IDF(t, d) = TF(t, d) × IDF(t)

A higher TF-IDF value generally indicates that a word is relatively important in a particular document compared with the entire collection.

5.4 TF-IDF in Scikit-learn

Scikit-learn’s TfidfVectorizer does not use exactly the same default formula as the simple textbook formula.

By default, it uses smoothed IDF:

IDF(t) = log((1 + N) / (1 + df(t))) + 1

It also applies L2 normalization by default.

These settings help avoid zero IDF values and normalize document vectors.

TF-IDF values are weights, not probabilities. They do not directly represent the probability that a word belongs to a document or category.

5.5 Historical Background

Karen Spärck Jones published an important research paper in 1972 titled A Statistical Interpretation of Term Specificity and Its Application in Retrieval.

Her work explained how the importance of a term can be estimated statistically using its frequency across a document collection. This became a foundation for the inverse document frequency idea used in information retrieval and TF-IDF.

6. Fitting, Transforming, and Data Leakage

When using text vectorizers, it is important to understand the difference between fit(), transform(), and fit_transform().

6.1 fit()

The fit() method learns the vocabulary and other information needed by the vectorizer from the training documents.

For example:

vectorizer.fit(training_documents)

This allows the vectorizer to learn which words will become features.

6.2 transform()

The transform() method converts documents into numerical vectors using the vocabulary that was already learned.

For example:

vectorizer.transform(test_documents)

It does not learn a new vocabulary.

6.3 fit_transform()

The fit_transform() method performs both operations:

1. Learns the vocabulary from the supplied documents.
2. Converts those documents into numerical vectors.

For example:

vectorizer.fit_transform(training_documents)

6.4 Out-of-Vocabulary (OOV) Words

An out-of-vocabulary word is a word that was not included in the vocabulary learned during fitting.

For example, suppose the training vocabulary is:

[machine, learning, data]

A test document contains:

“machine learning artificial intelligence”

The words “artificial” and “intelligence” were not learned during fitting. Standard Scikit-learn CountVectorizer and TfidfVectorizer ignore those unseen words during transformation.

6.5 Data Leakage

Data leakage happens when information from the test dataset is used during model training or preprocessing.

For example, fitting a vectorizer on the complete dataset before splitting it into training and testing data allows the vectorizer to learn vocabulary information from the test documents.

This can lead to an unrealistic evaluation of model performance.

The correct procedure is:

1. Split the dataset into training and testing sets.
2. Use fit_transform() on the training documents.
3. Use transform() on the testing documents.
4. Train the model using the training features.
5. Evaluate the model using the testing features.

This helps prevent test information from leaking into the training process.

7. Case Study: Vectorization and NLP Research for the Somali Language

Somali is a relatively low-resource language in Natural Language Processing. Compared with English, fewer digital datasets and language-processing resources are available.

This creates challenges for developing reliable text-processing and Machine Learning systems for Somali.

7.1 Somali Information Retrieval Corpus (2023)

A research paper titled Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources was published in the Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

The researchers were Abdisalam Mahamed Badel, Ting Zhong, Wenxin Tai, and Fan Zhou.

They created a Somali information retrieval corpus containing 2,335 documents collected from Somali online sources and publicly available cultural and historical texts.

The research reported:

* 2,335 documents.
* 578,216 words.
* 16,450 sentences.
* 16 search queries.

The researchers developed a Python-based indexing system that tokenized documents and created an inverted index using unigrams. The system also calculated term frequency (TF) and document frequency (DF).

They evaluated the corpus using a pseudo-relevance feedback technique, which expands a search query using important terms from initially retrieved documents.

This study demonstrates how vocabulary, tokenization, term frequency, and document frequency can support information retrieval for Somali text.

7.2 Somali Stop-Word Removal Research (2023)

Another research paper was published in the International Journal of Engineering Trends and Technology in December 2023.

Its title is Enhancing Natural Language Processing in Somali Text Classification: A Comprehensive Framework for Stop Word Removal.

The authors are Abdullahi Ahmed Abdirahman, Abdirahman Osman Hashi, Ubaid Mohamed Dahir, Mohamed Abdirahman Elmi, and Octavio Ernest Romo Rodriguez.

The study proposed a framework for identifying and removing Somali stop words. It discussed preprocessing steps such as:

* Noise identification and removal.
* Character normalization.
* Tokenization.
* Part-of-speech tagging.
* Lemmatization.

The researchers analyzed a dataset containing 79,741,231 tokens and 71,871,585 words.

This research is relevant to vectorization because stop-word removal and other preprocessing steps can change the vocabulary size and the resulting document-term matrix.

7.3 Comparison of Vectorization Techniques

The following techniques can be applied to Somali text:

One-Hot Encoding: Can represent individual Somali words numerically, but it does not capture relationships between words.

Bag of Words: Can count Somali word occurrences and create document-term matrices for classification and retrieval tasks.

N-grams: Can preserve short sequences of neighboring Somali words and capture useful phrases.

TF-IDF: Can assign higher weights to distinctive words and lower weights to words that occur in many documents.

The Somali research discussed above demonstrates the importance of building language-specific datasets and preprocessing resources before applying NLP techniques.

7.4 Conclusion

Text vectorization is an important step in Natural Language Processing because it converts unstructured text into numerical features that Machine Learning algorithms can process.

One-Hot Encoding, Bag of Words, N-grams, and TF-IDF provide different ways of representing text. Each technique has advantages and limitations.

For low-resource languages such as Somali, developing suitable datasets, preprocessing methods, and vocabulary resources is important for improving NLP applications.

⸻

REFERENCES

1. Scikit-learn. (2026). CountVectorizer.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
2. Scikit-learn. (2026). TfidfVectorizer.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
3. SciPy. Sparse matrices.
    https://docs.scipy.org/doc/scipy/reference/sparse.html
4. Spärck Jones, K. (1972). A Statistical Interpretation of Term Specificity and Its Application in Retrieval. Journal of Documentation, 28(1), 11–21.
    https://doi.org/10.1108/EB026526
5. Badel, A. M., Zhong, T., Tai, W., & Zhou, F. (2023). Somali Information Retrieval Corpus: Bridging the Gap between Query Translation and Dedicated Language Resources. Proceedings of EMNLP 2023, 7463–7469.
    https://aclanthology.org/2023.emnlp-main.462/
6. Abdirahman, A. A., Hashi, A. O., Dahir, U. M., Elmi, M. A., & Rodriguez, O. E. R. (2023). Enhancing Natural Language Processing in Somali Text Classification: A Comprehensive Framework for Stop Word Removal. International Journal of Engineering Trends and Technology, 71(12), 40–49.
    https://doi.org/10.14445/22315381/IJETT-V71I12P205
7. Scikit-learn. Common pitfalls and recommended practices.
    https://scikit-learn.org/stable/common_pitfalls.html



    # PART B – PRACTICAL

## 1. One-Hot Encoding

The vocabulary contains six words: cash, class, free, lunch, notes, and win. Each word is represented by a vector containing one 1 and five zeros. The vector length remains six regardless of the number of words in a sentence. A sentence with two words produces two vectors, while a sentence with three words produces three vectors.

## 2. Bag of Words (BoW)

The CountVectorizer vocabulary contains 32 unique words, and the document-term matrix has a shape of 5 × 32.

### Document-Term Matrix

The columns follow this vocabulary order:

```text
['are', 'before', 'but', 'call', 'cash', 'claim', 'class',
 'entry', 'for', 'free', 'good', 'great', 'lecture', 'lunch',
 'me', 'meeting', 'not', 'notes', 'now', 'phone', 'please',
 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today',
 'was', 'we', 'were', 'win']
```

| Document | Document-Term Vector |
|---|---|
| D1 | `[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,2]` |
| D2 | `[0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1]` |
| D3 | `[1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0]` |
| D4 | `[0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0]` |
| D5 | `[0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,2,0,0,1,0,1,0]` |

In D1, the word "win" has a value of 2 because it appears twice. The vectors for "win free phone" and "phone free win" are identical because Bag of Words counts word occurrences but ignores word order.

## 3. Tokens vs Vocabulary Size

- Total tokens using whitespace splitting: 42.
- Tokens recognized by CountVectorizer: 41.
- Vocabulary size: 32.
- Words appearing in more than one document: win, free, now, today, lecture, notes, and the.

The difference between 42 and 41 occurs because CountVectorizer ignores the one-character word "a" by default. Vocabulary size counts unique words, while total tokens count word occurrences.

## 4. Preprocessing Inside the Vectorizer

The default CountVectorizer converts text to lowercase, so "FREE", "Win", and "WIN" are treated as the same word. With `lowercase=False`, uppercase and lowercase forms remain separate features. The punctuation mark "!" is ignored, and the one-character word "a" is excluded by the default token pattern. When `stop_words="english"` is used, words such as "to" and "not" are removed. Removing "not" can be harmful in D5 because it changes the meaning of the phrase "not good".

## 5. Limiting the Vocabulary

The full vocabulary contains 32 unique words. With `min_df=2`, only words appearing in at least two different documents are retained, reducing the vocabulary to 7 words: free, lecture, notes, now, the, today, and win. With `max_features=5`, the vectorizer keeps five frequently occurring words: free, lecture, now, the, and win. These settings reduce the number of features, but they may remove useful words.

## 6. N-grams

Unigrams represent individual words, while bigrams represent pairs of consecutive words. In D5, the bigram "not good" preserves the reviewer's negative opinion by keeping the two words together. Across all five documents, the vocabulary size increases from 32 with unigrams to 67 with unigrams and bigrams, and to 98 with n-grams up to trigrams. This shows that larger n-gram ranges increase the number of features.

## 7. TF-IDF

TF-IDF assigns a weight to each word based on its frequency in a document and how rare it is across the collection.

### D2 – Word Counts and TF-IDF Weights

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

In D2, all words appear once, but their TF-IDF weights differ because they have different document frequencies. For example, "cash" has a higher weight than "free" because it appears in fewer documents. In D1, "win" has the highest weight because it appears twice in that document.

## 8. TF-IDF by Hand

The formulas used are:

- TF = Word Count / Total Tokens in the Document
- IDF = ln(Total Documents / Document Frequency)
- TF-IDF = TF × IDF

There are five documents in total, and D2 contains eight tokens.

### Calculation for "cash"

- Count = 1
- Total tokens in D2 = 8
- Document frequency = 1
- TF = 1 / 8 = 0.125
- IDF = ln(5 / 1) = 1.6094
- TF-IDF = 0.125 × 1.6094 = 0.2012

### Calculation for "today"

- Count = 1
- Total tokens in D2 = 8
- Document frequency = 2
- TF = 1 / 8 = 0.125
- IDF = ln(5 / 2) = 0.9163
- TF-IDF = 0.125 × 0.9163 = 0.1145

The manual calculation and Scikit-learn produce different numerical values because Scikit-learn uses smoothed IDF and L2 normalization by default. However, both methods rank "cash" higher than "today".

## 9. Sparse Matrix

The document-term matrix contains 160 cells in total. Only 39 cells contain non-zero values, while 121 cells contain zeros. Therefore, 75.62% of the matrix consists of zeros. Sparse matrix storage helps reduce memory usage by storing mainly non-zero values.

## 10. Fit vs Transform

The vectorizer learns its vocabulary using only the training documents (D1–D4). When transforming new documents, words that are not in the training vocabulary are treated as out-of-vocabulary (OOV) words.

- N1 OOV words: your, prize.
- N2 OOV words: is, moved, tomorrow.

The word "prize" is unknown because the training document contains "prizes". Stemming or lemmatization could help normalize different word forms.

## 11. Data Leakage

The training vocabulary contains 26 words, while the combined vocabulary contains 31 words. Five new words entered the vocabulary from the test documents: "is", "moved", "prize", "tomorrow", and "your". This demonstrates data leakage because information from the test set was used during vocabulary creation. To prevent leakage, the vectorizer should be fitted only on training data and then used to transform test data.

## 12. Vectorization Does Not Understand Meaning

The cosine similarity between R1 and R2 is 0.75, and the similarity between R1 and R3 is also 0.75. Although "tasty" and "delicious" have similar meanings, CountVectorizer does not understand synonyms or sentiment. It only compares shared words, such as "the", "food", and "was". This demonstrates that basic vectorization captures word occurrence rather than meaning.

## 13. Applying Vectorization to a Spam Detector

For an e-commerce spam detector, TF-IDF can help identify words that are important in spam messages, while Bag of Words can measure how often words appear. I would use unigrams and bigrams so the model can recognize individual words and useful phrases such as "free prize" or "claim now". I would avoid removing important words such as "not", because removing them can change the meaning of a message. Vocabulary limits such as `min_df` can reduce noise, but they should be selected carefully because rare words may be useful indicators of spam. Finally, the vectorizer must be fitted only on the training data to prevent data leakage.