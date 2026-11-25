## Part A – Research (Summarized Answers)

### 1. Text Vectorization and One-Hot Encoding

Text vectorization is the process of converting text into numerical representations that Machine Learning models can process.

Most traditional Machine Learning models require documents to have the same number of features because they perform mathematical operations on fixed-dimensional vectors.

One-hot encoding represents each word as a binary vector containing one 1 and zeros in all other positions. However, it represents individual words rather than complete documents and does not capture semantic relationships between words.

**Example:** In an email spam detector, words such as "win" and "free" can be converted into numerical features to help classify messages as spam or legitimate.

### 2. Bag of Words and Document-Term Matrix

Bag of Words (BoW) represents documents by counting how many times each vocabulary word appears, without considering word order.

A Document-Term Matrix contains:
- **Rows:** Documents.
- **Columns:** Unique vocabulary words.
- **Cells:** Word frequencies in each document.

It is called a "bag" because words are treated as an unordered collection.

Losing word order may have little effect when classifying news topics, but it causes problems in sentiment analysis. For example, "good" and "not good" have different meanings, even though they share similar words.

### 3. Vocabulary Size and Sparsity

Vocabulary size is the total number of unique words in a document collection.

Sparsity refers to the large proportion of zero values in a document-term matrix because each document contains only a small portion of the complete vocabulary.

Sparse matrices save memory by storing mainly non-zero values and their positions, for example, using Compressed Sparse Row (CSR) format.

Important vocabulary settings include:

| Setting | Function |
|---|---|
| min_df | Removes words appearing in too few documents. |
| max_df | Removes words appearing in too many documents. |
| max_features | Limits the number of selected vocabulary features. |

Preprocessing also changes vocabulary:
- Lowercasing combines uppercase and lowercase forms.
- Stopword removal removes common words.
- Stemming reduces related word forms to common stems.

### 4. N-grams

N-grams are consecutive sequences of words that preserve some word order.

- **Unigrams:** Individual words.
- **Bigrams:** Two consecutive words.
- **Trigrams:** Three consecutive words.

Example: "not good" is a bigram that communicates negative sentiment.

Larger n-grams provide more context but increase vocabulary size, memory usage, computational cost, and the risk of overfitting.

### 5. TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) measures how important a word is in a document compared with the entire document collection.

It combines:

- **TF:** How frequently a word appears in a document.
- **IDF:** How rare the word is across all documents.


It also applies L2 normalization by default. TF-IDF values are weights, not probabilities, because they measure relative feature importance rather than probability distributions.

### 6. Fit, Transform, OOV and Data Leakage

The three main vectorizer operations are:

- **fit():** Learns the vocabulary and, for TF-IDF, IDF weights from training documents.
- **transform():** Converts documents using the already learned vocabulary.
- **fit_transform():** Learns the vocabulary and transforms the same documents in one operation.

Out-of-vocabulary (OOV) words are unknown words that are ignored during transformation.

Data leakage occurs when information from test documents is used during training or preprocessing, making evaluation results unreliable.

The correct procedure is to split the data first, fit the vectorizer only on training data, transform both training and test data, train the model, and finally evaluate it on the test data.

### 7. Case Study: Amharic Sentiment Classification

Neshir, Rauber, and Atnafu (2021) studied sentiment classification in Amharic, an African language spoken mainly in Ethiopia.

The researchers used four datasets containing 13,392 sentiment-labelled texts and compared TF-IDF word unigrams with TF-IDF character n-grams.

Their stacking classifier achieved 79% accuracy using character n-grams, compared with 73% using word unigrams in the reported experiment without SMOTE.

The study demonstrates that different text representations can produce different classification results. Character n-grams can capture spelling patterns and word variations, but neither ordinary word-frequency features nor character patterns fully understand semantic meaning.

For example, "good" and "excellent" may express similar opinions but are treated as different features by basic TF-IDF.

This shows why more advanced representations, such as word embeddings and contextual language models, are useful for capturing semantic relationships.



# Part B – Practical: Text Vectorization

## Task 2: Document-Term Matrix

The following table shows the complete Document-Term Matrix generated using `CountVectorizer` on D1–D5. Rows represent documents, columns represent vocabulary words, and cells represent word frequencies.

**Matrix shape: (5, 32)**

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |

| Document | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

**Explanation:** The cell containing 2 in D1 is the word `win` because it appears twice in that document.

## Task 7: D2 Count and TF-IDF Comparison

The following table compares the word counts and TF-IDF weights for every word in D2.

| Word | Count | TF-IDF Weight |
|---|---:|---:|
| call | 1 | 0.3891 |
| cash | 1 | 0.3891 |
| entry | 1 | 0.3891 |
| free | 1 | 0.3140 |
| now | 1 | 0.3140 |
| prizes | 1 | 0.3891 |
| today | 1 | 0.3140 |
| win | 1 | 0.3140 |

**Explanation:** Every word in D2 appears once, but their TF-IDF weights are different because they appear in different numbers of documents. Words appearing in fewer documents receive higher IDF weights. Therefore, `cash`, `call`, `entry`, and `prizes` receive higher weights than `free`, `now`, `today`, and `win`.

## Task 8: TF-IDF Calculations by Hand

The textbook formula is:

\[
TF\text{-}IDF=TF\times IDF
\]

\[
IDF=\log\left(\frac{N}{df}\right)
\]

Where:
- N = Total number of documents.
- df = Number of documents containing the word.
- TF = Number of times the word appears in D2.

There are five documents, so N = 5.

**1. Word: cash**

- TF = 1
- N = 5
- df = 1

\[
IDF=\log(5/1)=1.6094
\]

\[
TF\text{-}IDF=1\times1.6094
\]

\[
\boxed{TF\text{-}IDF=1.6094}
\]

**2. Word: today**

- TF = 1
- N = 5
- df = 2

\[
IDF=\log(5/2)=0.9163
\]

\[
TF\text{-}IDF=1\times0.9163
\]

\[
\boxed{TF\text{-}IDF=0.9163}
\]

**Comparison:** Cash has a higher TF-IDF score than today because it appears in fewer documents. Both the manual calculations and scikit-learn rank cash higher than today. However, their numerical values differ because scikit-learn uses smoothed IDF and L2 normalization by default.

## Short Answers to Practical Questions

### Task 1: One-Hot Encoding

Each word receives a separate vector based on its position in the vocabulary. The vector for `free` is `[0, 0, 1, 0, 0, 0]`, while the vector for `cash` is `[1, 0, 0, 0, 0, 0]`. The number of vectors depends on the number of words in the sentence because each word receives its own vector.

### Task 2: Bag of Words

The Document-Term Matrix has five rows and 32 columns, representing five documents and 32 unique words. The word `win` has a count of 2 in D1 because it appears twice. The vectors for `win free phone` and `phone free win` are identical because Bag of Words ignores word order and counts word frequencies.

### Task 4: Preprocessing Inside the Vectorizer

The default vectorizer converts uppercase words into lowercase, so `Win` and `WIN` become `win`. Punctuation such as `!` is ignored, and the single-character word `a` is excluded by the default token pattern. The English stopword list removes words such as `a` and `to`, and it also removes `not` from D5, which is dangerous because it changes the meaning of the negative phrase `not good`.

### Task 5: Limiting the Vocabulary

With `min_df=2`, the vocabulary contains seven words: `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win`. With `max_features=5`, only five frequent words remain: `free`, `lecture`, `now`, `the`, and `win`. These settings remove less frequent words and reduce vocabulary size, but they may also remove useful information.

### Task 6: N-Grams

The unigram vocabulary represents individual words, while the bigram vocabulary also represents pairs of consecutive words. The bigram `not good` is important because it preserves the negative opinion in D5. Across D1–D5, vocabulary size increases from 32 unigrams to 67 unigram-and-bigram features and 98 features when trigrams are included.

### Task 7: TF-IDF

TF-IDF gives more importance to words that are frequent in a particular document but less common across the entire collection. In D2, `cash` receives a higher weight than `today` because it appears in only one document. In D1, `win` receives the highest weight because it appears twice and is relatively uncommon across the corpus.

### Task 8: TF-IDF by Hand

The manual TF-IDF scores are 1.6094 for `cash` and 0.9163 for `today`. This means `cash` is more informative because it appears in fewer documents. The ranking is the same as scikit-learn, although the numerical weights differ because scikit-learn applies IDF smoothing and vector normalization.

### Task 10: Fit and Transform

A vectorizer should first learn its vocabulary from the training documents D1–D4 and then transform new documents using that vocabulary. The word `prize` is unknown because the training vocabulary contains `prizes`, which is a different word form. Stemming or lemmatization can help convert related word forms into a common base form.

### Task 11: Data Leakage

Data leakage happens when information from test data is used during training. When the vectorizer is fitted on D1–D4 only, it does not know the test-only words `is`, `moved`, `prize`, `tomorrow`, and `your`. These five words enter the vocabulary when the test documents are incorrectly included during fitting. Therefore, the vectorizer must be fitted on training data only.

### Task 12: No Meaning – Cosine Similarity

The cosine similarity between R1 and R2 is 0.75, and the similarity between R1 and R3 is also 0.75. R1 is therefore equally similar to both reviews according to Bag of Words. This happens because the vectorizer recognizes shared words but cannot understand that `tasty` and `delicious` have similar meanings while `terrible` has a negative meaning.

## Final Discussion: Choosing Vectorizer Settings for a Spam Detector

For a spam detector, I would start with `CountVectorizer` as a baseline and also test `TfidfVectorizer` to compare their performance. I would keep lowercase conversion enabled and avoid removing all English stopwords because words such as `not` can change the meaning of a message. I would use unigrams and bigrams because individual words such as `free`, `win`, and `cash`, together with phrases such as `claim now`, can help identify spam messages. I would be careful with `min_df=2` because important spam words such as `cash` and `claim` may appear in only one document, as shown by the practical vocabulary. I would also avoid setting `max_features=5` because it removes too much useful information. Finally, I would fit the vectorizer on training data only to prevent data leakage and evaluate both approaches using unseen test data.