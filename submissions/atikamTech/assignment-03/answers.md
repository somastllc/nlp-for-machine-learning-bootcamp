# Assignment 3 – Vectorization

**Student:** Atika Isse Ali  
**GitHub:** atikamTech  
**Course:** NLP for Machine Learning Bootcamp

# Part A: Research

## 1. What is Vectorization?

Vectorization is the process of converting text into numerical representations that machine learning algorithms can understand. Computers cannot directly learn from ordinary words, so vectorization changes words or documents into numbers.

A vocabulary is created, and each word is assigned a position in a numerical vector. Documents must use the same vocabulary and therefore have the same number of features.

**One-hot encoding** represents a word using a vector containing zeros and one 1. For example, with the vocabulary `[cash, free, win]`:

- cash = `[1, 0, 0]`
- free = `[0, 1, 0]`
- win = `[0, 0, 1]`

One-hot encoding is not enough for representing a complete document because it represents individual words separately. A document with many words needs multiple vectors or another representation.

**Real-life example:** A spam detection system can convert email words into numerical features and use those features to identify suspicious messages.

## 2. Bag of Words and Document-Term Matrix

Bag of Words (BoW) represents a document by counting how often each vocabulary word appears. It ignores grammar and word order.

A Document-Term Matrix (DTM) organizes these counts into a table:

- **Rows:** Documents.
- **Columns:** Vocabulary words.
- **Cells:** Number of times each word appears in a document.

It is called a bag of words because the method treats the document as a collection of words rather than considering their original order.

**Example:** The sentences "win free phone" and "phone free win" have the same BoW representation.

Word order does not matter for a basic topic or keyword-counting task. However, it matters for sentiment analysis. For example, "good" and "not good" have different meanings, but a simple BoW representation may not fully capture that difference.

## 3. Vocabulary Size and Sparsity

Vocabulary size is the number of unique words in the dataset. As more documents are added, the vocabulary can become very large.

A document-term matrix is often sparse because each document uses only a small portion of the complete vocabulary. Most cells contain zero.

Instead of storing every zero, sparse matrix formats store mainly non-zero values and their positions. This saves memory and can improve computational efficiency.

Three useful CountVectorizer parameters are:

- `min_df`: Removes words that appear in fewer than the specified number or proportion of documents.
- `max_df`: Removes words that appear in more than the specified number or proportion of documents.
- `max_features`: Limits the vocabulary to the most frequent features.

Preprocessing also changes vocabulary size. Lowercasing can combine uppercase and lowercase versions of a word, while stop-word removal, stemming, and lemmatization can reduce the number of different tokens.

## 4. N-grams

N-grams are sequences of consecutive words used as features.

- **Unigram:** One word, such as `good`.
- **Bigram:** Two consecutive words, such as `not good`.
- **Trigram:** Three consecutive words, such as `this is good`.

N-grams help preserve some word-order information that a unigram model loses.

For example, `good` and `not good` express different opinions. A bigram feature can represent `not good` as a separate feature.

However, larger n-grams increase the vocabulary size and memory requirements. They can also create many features that appear only rarely.

## 5. TF-IDF

TF-IDF means Term Frequency–Inverse Document Frequency. It gives a numerical weight to words based on how frequently they appear in a document and how common they are across all documents.

**Term Frequency (TF)** measures how often a term occurs in a document.

**Inverse Document Frequency (IDF)** measures how rare a term is across the collection of documents.

A textbook formula is:

\[
TF(t,d)=\frac{\text{count of }t\text{ in }d}{\text{total terms in }d}
\]

\[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
\]

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

Here, \(N\) is the total number of documents and \(df(t)\) is the number of documents containing the term.

A rare word receives a higher IDF because it can help distinguish one document from others. A very common word receives a lower IDF.

Scikit-learn's `TfidfVectorizer` uses a smoothed IDF formula by default:

\[
IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

It also applies L2 normalization by default. Therefore, its final weights can differ from the unnormalized textbook calculation.

TF-IDF values are weights, not probabilities. A weight does not represent the probability that a word belongs to a document.

## 6. fit(), transform(), and fit_transform()

These methods are important when preparing text for machine learning.

- `fit()` learns the vocabulary and other information from training documents.
- `transform()` converts new documents using the vocabulary already learned.
- `fit_transform()` learns the vocabulary and transforms the training documents in one step.

When a new text contains a word that was not in the training vocabulary, that word is out of vocabulary (OOV) and is normally ignored by CountVectorizer.

**Data leakage** happens when information from the test set is used during training. For example, fitting a vectorizer on both training and test documents allows the model's vocabulary to include words from the test set.

The correct process is to fit the vectorizer on training data only, then transform both training and test data using that fitted vectorizer.

## 7. African-Language NLP Case Study

Keinan and Hacohen-Kerner (2023) studied sentiment analysis for tweets written in low-resource African languages in SemEval-2023 Tasks 12A and 12B.

Their work explored machine learning and deep learning approaches and used word and character n-gram features weighted with TF-IDF. The study is relevant because many African languages have limited NLP resources compared with languages such as English.

This case study shows how vectorization can help convert social media text into features for sentiment classification. Word n-grams can represent individual words and short phrases, while character n-grams can capture parts of words and spelling variations.

However, traditional BoW and TF-IDF features do not fully understand meaning or context. Two words can have similar meanings but be treated as completely different features.

For Somali NLP, this is an important challenge because spelling differences, dialects, and limited language resources can affect text processing. More research and language-specific preprocessing are needed to improve results.

# Part B: Practical Work

**Tools used:** Python, NumPy, and scikit-learn.

The following experiments use the five training documents and two test texts provided in the assignment.

## Task 1: One-Hot Encoding

Vocabulary:

`['cash', 'class', 'free', 'lunch', 'notes', 'win']`

| Word | One-hot vector |
|---|---|
| free | `[0, 0, 1, 0, 0, 0]` |
| cash | `[1, 0, 0, 0, 0, 0]` |
| win | `[0, 0, 0, 0, 0, 1]` |

For the sentence `win free cash`, the program produced three vectors.

- Number of vectors: 3
- Length of each vector: 6

**Answer:** Each word gets its own vector, and the vector length depends on the vocabulary size. The number of vectors depends on the number of words in the sentence. This method represents words separately and does not capture their meanings or relationships.

## Task 2: Bag of Words and Document-Term Matrix

The CountVectorizer vocabulary contains 32 features.

**Vocabulary order:**

`['are', 'before', 'but', 'call', 'cash', 'claim', 'class', 'entry', 'for', 'free', 'good', 'great', 'lecture', 'lunch', 'me', 'meeting', 'not', 'notes', 'now', 'phone', 'please', 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today', 'was', 'we', 'were', 'win']`

**Matrix shape:** 5 × 32

The following matrix uses the vocabulary order above. Each row represents D1 through D5.

| Document | Values in vocabulary order |
|---|---|
| D1 | `[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,2]` |
| D2 | `[0,0,0,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1]` |
| D3 | `[1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0]` |
| D4 | `[0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0]` |
| D5 | `[0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,2,0,0,1,0,1,0]` |

**D1 word counts:**

| Word | Count |
|---|---:|
| claim | 1 |
| free | 1 |
| now | 1 |
| phone | 1 |
| reply | 1 |
| to | 1 |
| win | 2 |

**Order comparison:**

- `win free phone` produced the same vector as `phone free win`.
- The program confirmed: `Are the vectors equal? True`

**Answer:** The word `win` has a count of 2 in D1 because it appears twice, including the uppercase version `WIN`, which is converted to lowercase by default. The two sentences have equal vectors because Bag of Words counts words without considering their order.

## Task 3: Tokens and Vocabulary Size

| Document | Token occurrences |
|---|---:|
| D1 | 8 |
| D2 | 8 |
| D3 | 7 |
| D4 | 8 |
| D5 | 10 |
| **Total** | **41** |

- Total token occurrences: 41
- Vocabulary size: 32
- Words appearing in more than one document: `free`, `lecture`, `notes`, `now`, `the`, `today`, `win`.

There are 41 total token occurrences but only 32 unique vocabulary words because some words appear more than once, either in the same document or in different documents.

## Task 4: Preprocessing Inside the Vectorizer

**Default vocabulary:**

`['claim', 'free', 'now', 'phone', 'reply', 'to', 'win']`

**Vocabulary with `lowercase=False`:**

`['FREE', 'Reply', 'WIN', 'Win', 'claim', 'now', 'phone', 'to']`

**Vocabulary with `stop_words='english'`:**

`['claim', 'free', 'phone', 'reply', 'win']`

**English stop words on D1–D5:**

`['cash', 'claim', 'class', 'entry', 'free', 'good', 'great', 'lecture', 'lunch', 'meeting', 'notes', 'phone', 'prizes', 'reply', 'send', 'today', 'win']`

The word `not` is not in this vocabulary.

**Answer:** By default, CountVectorizer converts words to lowercase, so `Win` and `WIN` become the same feature. With `lowercase=False`, uppercase and lowercase forms remain separate. Punctuation is removed, and the default token pattern excludes one-letter words such as `a`. English stop-word removal removes common words such as `to`, but it also removes `not`, which is important in sentiment analysis. This shows that removing stop words is not always suitable for every NLP task.

## Task 5: Limiting the Vocabulary

| Setting | Vocabulary size | Vocabulary |
|---|---:|---|
| Full vocabulary | 32 | All 32 features from Task 2 |
| `min_df=2` | 7 | `free`, `lecture`, `notes`, `now`, `the`, `today`, `win` |
| `max_features=5` | 5 | `free`, `lecture`, `now`, `the`, `win` |

**Answer:** `min_df=2` keeps words that appear in at least two documents. This reduces the vocabulary from 32 to 7 features. `max_features=5` limits the vocabulary to five selected features. These settings reduce the number of features, but they can also remove useful words that appear less frequently.

## Task 6: N-grams

**D5 unigram vocabulary:**

`['but', 'good', 'great', 'lecture', 'not', 'notes', 'the', 'was', 'were']`

**D5 unigram and bigram vocabulary:**

`['but', 'but the', 'good', 'good but', 'great', 'lecture', 'lecture was', 'not', 'not good', 'notes', 'notes were', 'the', 'the lecture', 'the notes', 'was', 'was not', 'were', 'were great']`

**N-gram vocabulary sizes:**

| N-gram range | Number of features |
|---|---:|
| `(1, 1)` | 32 |
| `(1, 2)` | 67 |
| `(1, 3)` | 98 |

**Answer:** The bigram `not good` preserves a phrase that expresses a negative opinion. Unigrams alone represent `not` and `good` separately. The vocabulary grows from 32 to 67 and then 98 features as larger n-grams are added. Larger n-grams can preserve more word-order information, but they also increase memory and processing requirements.

## Task 7: TF-IDF

The TF-IDF matrix has the same shape as the CountVectorizer matrix: 5 × 32.

**D1 word counts and TF-IDF weights:**

| Word | Count | TF-IDF weight |
|---|---:|---:|
| claim | 1 | 0.3557 |
| free | 1 | 0.2869 |
| now | 1 | 0.2869 |
| phone | 1 | 0.3557 |
| reply | 1 | 0.3557 |
| to | 1 | 0.3557 |
| win | 2 | 0.5739 |

**D2 word counts and TF-IDF weights:**

| Word | Count | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.3891 |
| cash | 1 | 0.3891 |
| entry | 1 | 0.3891 |
| free | 1 | 0.3140 |
| now | 1 | 0.3140 |
| prizes | 1 | 0.3891 |
| today | 1 | 0.3140 |
| win | 1 | 0.3140 |

**Answer:** In D1, `win` has the highest weight (0.5739) because it appears twice. In D2, words such as `cash`, `entry`, and `prizes` have higher weights because they occur in fewer documents than common words such as `free`, `now`, `today`, and `win`. TF-IDF gives more importance to words that help distinguish a document from the others. The displayed scikit-learn values are also L2-normalized.

## Task 8: TF-IDF by Hand

The calculation uses D2 and the textbook formula.

- Total documents, N = 5
- D2 analyzed tokens = 8
- D2 tokens: `free`, `entry`, `win`, `cash`, `prizes`, `today`, `call`, `now`

### Word: cash

- Count in D2 = 1
- Document frequency = 1
- TF = 1 / 8 = 0.125000
- IDF = log(5 / 1) = 1.609438
- Textbook TF-IDF = 0.125000 × 1.609438 = **0.201180**
- Scikit-learn normalized TF-IDF = **0.389141**

### Word: today

- Count in D2 = 1
- Document frequency = 2
- TF = 1 / 8 = 0.125000
- IDF = log(5 / 2) = 0.916291
- Textbook TF-IDF = 0.125000 × 0.916291 = **0.114536**
- Scikit-learn normalized TF-IDF = **0.313957**

**Answer:** `cash` has a higher textbook TF-IDF weight than `today` because it appears in only one document, while `today` appears in two documents. This makes `cash` more useful for distinguishing D2. Scikit-learn uses smoothed IDF and L2 normalization, so its final weights are different from the textbook values, but the ranking is the same.

## Task 9: Sparse Matrix

| Measurement | Result |
|---|---:|
| Matrix shape | 5 × 32 |
| Total cells | 160 |
| Non-zero cells | 39 |
| Zero cells | 121 |
| Percentage of zeros | 75.62% |

The matrix contains 121 zero cells out of 160 total cells.

This demonstrates sparsity: most vocabulary words do not appear in most documents. Sparse matrix storage is useful because it avoids storing every zero value.

## Task 10: Fit and Transform

The vectorizer was fitted using D1–D4 only.

**Training vocabulary:**

`['are', 'before', 'call', 'cash', 'claim', 'class', 'entry', 'for', 'free', 'lecture', 'lunch', 'me', 'meeting', 'notes', 'now', 'phone', 'please', 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today', 'we', 'win']`

**Test text 1:** `Claim your free prize now`

Vector:

`[0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]`

**Test text 2:** `Is the lunch meeting moved to tomorrow?`

Vector:

`[0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0]`

**Out-of-vocabulary words:**

- N1: `prize`, `your`
- N2: `is`, `moved`, `tomorrow`

**Answer:** Words outside the training vocabulary are ignored during transformation. The word `prize` is unknown even though D2 contains `prizes`, because they are different word forms. Stemming or lemmatization could help normalize words such as `prize` and `prizes` to a common form before vectorization.

## Task 11: Data Leakage

**Training-only vocabulary:** 26 features.

**Vocabulary after fitting on training and test texts:** 31 features.

**Words introduced by the test set:**

`['is', 'moved', 'prize', 'tomorrow', 'your']`

**Answer:** When the vectorizer is fitted on training and test texts together, it learns five additional words from the test set. This is data leakage because information from the test data is being used during the training preparation stage. The correct approach is to fit the vectorizer on training documents only and then transform the test documents using that same fitted vectorizer.

## Task 12: Vectors Do Not Capture Meaning

**Review vocabulary:**

`['delicious', 'food', 'tasty', 'terrible', 'the', 'was']`

**Review vectors:**

| Review | Vector |
|---|---|
| R1: the food was tasty | `[0, 1, 1, 0, 1, 1]` |
| R2: the food was delicious | `[1, 1, 0, 0, 1, 1]` |
| R3: the food was terrible | `[0, 1, 0, 1, 1, 1]` |

**Cosine similarity matrix:**

| | R1 | R2 | R3 |
|---|---:|---:|---:|
| R1 | 1.00 | 0.75 | 0.75 |
| R2 | 0.75 | 1.00 | 0.75 |
| R3 | 0.75 | 0.75 | 1.00 |

**Answer:** R1 (`tasty`) has the same similarity score of 0.75 with R2 (`delicious`) and R3 (`terrible`). This happens because all three reviews share the words `the`, `food`, and `was`. The model does not understand that `tasty` and `delicious` are positive words while `terrible` is negative. This demonstrates a limitation of simple Bag of Words representations.

# Task 13: Recommendation for a Spam Detector

For a basic spam detection system, I would start with TF-IDF because it gives more importance to words that distinguish spam messages from other messages. CountVectorizer is also useful as a baseline because it shows the number of times words appear. I would test unigrams and bigrams because phrases such as `free phone` and `claim now` can provide useful information. I would be careful when removing stop words because words such as `not` can change the meaning of a message. I would also consider `min_df` and `max_features` to control the vocabulary size. In this experiment, the matrix had 75.62% zeros, showing why sparse representations are useful. I would compare the results of these settings using a separate validation set before choosing the final model.

# References

1. Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM, 18*(11), 613–620. https://doi.org/10.1145/361219.361220

2. Scikit-learn. (n.d.). *CountVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

3. Scikit-learn. (n.d.). *TfidfVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

4. Scikit-learn. (n.d.). *Feature extraction: Text feature extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html

5. Keinan, R., & Hacohen-Kerner, Y. (2023). JCT at SemEval-2023 Tasks 12 A and 12B: Sentiment Analysis for Tweets Written in Low-resource African Languages using Various Machine Learning and Deep Learning Methods, Resampling, and HyperParameter Tuning. *Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)*. https://doi.org/10.18653/v1/2023.semeval-1.49

6. Abdirahman, et al. (2023). Comparative Analysis of Machine Learning and Deep Learning Models for Sentiment Analysis in Somali Language. *SSRG International Journal of Electrical and Electronics Engineering, 10*(7), 41–52. https://doi.org/10.14445/23488379/IJEEE-V10I7P104
