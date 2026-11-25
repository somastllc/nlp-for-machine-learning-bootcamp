# Assignment 3: Vectorization

## Part A: Research

### 1. Text Vectorization

Text vectorization is the process of converting text into numerical features that can be used by Machine Learning algorithms. Most traditional Machine Learning models require a fixed number of input features for every example, so documents with different lengths cannot simply be passed to the model as raw text. Vectorization solves this problem by defining a vocabulary and representing every document using the same feature positions.

One simple method is one-hot encoding. Each word in a vocabulary is assigned its own position, and the vector contains `1` at that word's position and `0` at all other positions. For example, if a system has the vocabulary `[cash, free, phone, prize]`, the word `free` can be represented as `[0, 1, 0, 0]`. However, one-hot encoding a whole document produces one vector for each word, so documents with different numbers of words produce different numbers of vectors. It also does not represent relationships or similarity between words.

A real-life example is an email spam detector. Words such as `prize`, `cash`, and `free` can be represented numerically and then used as features by a classification model. However, the one-hot representation alone cannot tell the model that words such as `excellent` and `great` have related meanings.

---

### 2. Bag of Words and the Document-Term Matrix

Bag of Words (BoW) represents a document using the number of times each vocabulary word appears in it. After processing several documents, the individual vectors are arranged into a document-term matrix.

The rows represent documents, while the columns represent vocabulary terms. Each cell contains the number of times a particular word occurs in that document. The matrix can therefore be used as a fixed-size numerical representation of the documents.

It is called a "bag" because the representation does not preserve the original order of the words. For example, `free prize` and `prize free` contain the same words and therefore receive the same Bag of Words representation.

Losing word order is often acceptable for tasks such as basic spam classification, where individual words such as `cash`, `prize`, or `winner` can be useful indicators. However, it can cause problems in sentiment analysis because `not good` and `good` contain the same words but express different meanings. N-grams can partly address this problem by keeping short sequences of neighboring words.

---

### 3. Vocabulary Size and Sparsity

Vocabulary size is the number of unique terms represented by the vectorizer. In a document-term matrix, it is therefore the number of columns. A token is an occurrence of a word, while a vocabulary contains unique terms, so repeated words increase the token count but do not create additional vocabulary columns.

Real datasets can contain thousands or millions of different terms. When these terms become columns, each document usually contains only a small fraction of the complete vocabulary. Most cells therefore contain zero, producing a sparse matrix.

Sparse matrices are stored efficiently by keeping information about non-zero entries rather than storing every zero explicitly. Scikit-learn's `CountVectorizer` produces a sparse representation of token counts.

The vocabulary can be controlled using parameters such as `min_df`, `max_df`, and `max_features`. `min_df` removes terms that occur in too few documents, while `max_df` can remove terms that occur in too many documents. `max_features` limits the number of selected features.

Preprocessing also changes vocabulary size. Lowercasing can combine words such as `Free` and `free`, while stemming can combine related forms such as `connect`, `connected`, and `connecting`. Stopword removal can remove common words. These choices can reduce the number of features, but they can also remove information that is useful for a particular task.

---

### 4. N-grams

An n-gram is a sequence of `n` neighboring words. A unigram contains one word, a bigram contains two consecutive words, and a trigram contains three consecutive words.

For example, the sentence:

`the movie was not good`

contains the bigram `not good`. A unigram representation contains `not` and `good` separately, while a bigram can preserve the relationship between them. This is useful because `not good` expresses a different sentiment from `good`.

N-grams therefore preserve some word order that ordinary Bag of Words loses. However, increasing the maximum n-gram size can greatly increase the vocabulary because every new word sequence can become a new feature. This increases memory requirements and can create many rare features.

For this reason, n-grams are usually selected carefully according to the task and the amount of available training data.

---

### 5. TF-IDF

TF-IDF stands for Term Frequency-Inverse Document Frequency. It gives a numerical weight to a word based on two ideas: how often the word occurs in a particular document and how common the word is across the document collection.

Term Frequency measures how frequently a term occurs in a document:

\[
TF(t,d)=\frac{\text{count of term }t\text{ in document }d}{\text{number of terms in document }d}
\]

The traditional inverse document frequency is:

\[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
\]

where `N` is the total number of documents and `df(t)` is the number of documents containing the term.

The basic TF-IDF formula is:

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

The idea of weighting terms according to how specific they are to documents is strongly associated with Karen Spärck Jones's 1972 work on term specificity. She argued that terms occurring in fewer documents can be more useful for distinguishing documents than terms occurring everywhere.

For example, if `the` appears in almost every document, it provides little information for distinguishing one document from another. A word such as `scholarship` that occurs in only a few documents can be more informative, so it receives a higher IDF value.

Scikit-learn does not use exactly the simple textbook formula by default. Its default smoothed IDF is:

\[
IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

It also applies L2 normalization by default. Smoothing prevents problems with document-frequency values, while normalization makes each output vector have unit Euclidean length.

TF-IDF values are weights, not probabilities. A value such as `0.55` does not mean that there is a 55% probability that a word is present or that a document belongs to a particular class. It is simply the numerical importance assigned to that term in the vector representation.

---

### 6. Fit, Transform, and Data Leakage

`fit` means learning information from the input text. For `CountVectorizer`, fitting learns the vocabulary. For a TF-IDF vectorizer, fitting also learns the document-frequency information needed for IDF.

`transform` uses the vocabulary that was already learned during fitting and converts new text into vectors. It does not create new vocabulary columns. `fit_transform` performs both operations together on the same data. Scikit-learn describes `fit_transform` as learning the vocabulary and returning the document-term matrix, while `transform` extracts counts using the vocabulary learned by `fit`.

If a new document contains a word that was not present when the vectorizer was fitted, that word is out of vocabulary (OOV). It has no column in the learned representation, so it is ignored during transformation.

Data leakage happens when information from the test set is allowed to influence the training process. In vectorization, one example is fitting the vectorizer on both training and test text before evaluation. This allows words from the test set to enter the vocabulary and can produce an overly optimistic evaluation. Scikit-learn recommends splitting the data first, fitting preprocessing only on the training data, and then transforming both training and test data using the learned transformation.

The correct order is:

```text
Split data
    ↓
Fit vectorizer on training data
    ↓
Transform training data
    ↓
Transform test data using the same vectorizer
```

The test data should not be used during fitting.

---

### 7. Case Study: Vector Representations for Amharic Idiom Recognition

A useful African-language case study is the 2023 study **"Deep learning-based idiomatic expression recognition for the Amharic language"** by Demeke Endalie, Getamesay Haile, and Wondmagegn Taye.

The study focuses on recognizing idiomatic expressions in Amharic. The researchers created a dataset containing 1,700 idiomatic expressions and 1,600 non-idiomatic expressions collected from Amharic books. They proposed a CNN model using FastText word embeddings and evaluated it using an 80:10:10 train, validation, and test split.

The study also compared different word representations. The reported comparison included FastText, TF-IDF, and one-hot encoding. CNN with FastText achieved 80% recognition accuracy, compared with 74% using TF-IDF and 71.3% using one-hot encoding. The authors explain the stronger result of FastText by its ability to preserve contextual information and represent the meaning of the expressions better than the traditional representations.

This case study illustrates an important limitation of Bag of Words and TF-IDF. These methods mainly represent whether particular words or word patterns occur and assign numerical weights to them. They do not create a semantic representation in which words or expressions with related meanings automatically receive similar representations. This becomes especially important for idioms, because their meaning cannot always be understood simply by considering the individual words.

The study therefore supports the idea that traditional representations such as TF-IDF can be useful baselines, while embedding-based representations can provide richer information about meaning and context.

---

# Part B: Practical

## 1. One-Hot Encoding

The one-hot encoder gives each vocabulary word its own position. Each word therefore produces one vector, so a sentence with three words produces three vectors, while a sentence with five words produces five vectors. This shows why one-hot encoding alone is not a fixed-size representation for complete documents.

## 2. Bag of Words

The `CountVectorizer` produced a document-term matrix with one row for each of the five documents and one column for each vocabulary word. The value `2` in D1 occurs under `win` because `Win` and `WIN` both become `win` after the vectorizer's default lowercasing. The vectors for `win free phone` and `phone free win` are identical because Bag of Words does not preserve word order.

### Document-Term Matrix

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

## 3. Tokens vs Vocabulary Size

The corpus contains 41 tokens according to the vectorizer's analyzer and has a vocabulary size of 32. The words appearing in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win`. Repeated occurrences increase the token count, while repeated vocabulary words still occupy only one column.

## 4. Preprocessing Inside the Vectorizer

The default vectorizer lowercases the text and removes punctuation, so `Win` and `WIN` become the same feature, while `!` disappears. The one-letter word `a` is not included by the default token pattern, and the English stopword list removes words such as `a`, `now`, and `to`. When the English stopword list is applied to D1-D5, `not` is also removed, which can destroy the negative meaning in D5.

## 5. Limiting the Vocabulary

With `min_df=2`, the surviving words are `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win`, because these occur in at least two documents. With `max_features=5`, only five high-frequency features are retained. These settings reduce the number of columns, but they can also remove less frequent words that may still be useful for classification.

## 6. N-grams

The bigram `not good` preserves the negative opinion in D5 better than unigrams alone because it keeps the relationship between the two words. The vocabulary grows from 32 unigrams to 67 features when bigrams are added and to 98 features when trigrams are also added. This shows that larger n-gram ranges can increase the number of features quickly.

## 7. TF-IDF

TF-IDF gives different weights to words even when their counts are the same because it considers how many documents contain each word. In D2, words such as `cash`, `call`, `entry`, and `prizes` receive higher weights because they are less common across the corpus. In D1, `win` has the highest weight because it occurs twice and therefore has a larger term-frequency contribution.

### D2 Count and TF-IDF

| Word | Count | TF-IDF |
|---|---:|---:|
| call | 1 | 0.389 |
| cash | 1 | 0.389 |
| entry | 1 | 0.389 |
| free | 1 | 0.314 |
| now | 1 | 0.314 |
| prizes | 1 | 0.389 |
| today | 1 | 0.314 |
| win | 1 | 0.314 |

## 8. TF-IDF by Hand

D2 contains eight tokens according to the vectorizer. For `cash`, the count is 1 and its document frequency is 1, so:

```text
TF = 1 / 8 = 0.125

IDF = log(5 / 1)
    = 1.609

TF-IDF = 0.125 × 1.609
       ≈ 0.201
```

For `today`, the count is 1 and its document frequency is 2 because it appears in D2 and D3:

```text
TF = 1 / 8 = 0.125

IDF = log(5 / 2)
    ≈ 0.916

TF-IDF = 0.125 × 0.916
       ≈ 0.115
```

The hand calculation ranks `cash` above `today`, which is consistent with the scikit-learn result. The exact numerical values differ because scikit-learn uses a smoothed IDF formula and normalizes the final vectors.

## 9. Sparse Matrix

The count matrix contains 160 cells, of which 39 are non-zero and 121 are zero. Therefore, approximately 75.63% of the matrix is zero. This demonstrates why text document-term matrices are usually sparse and why sparse storage is useful.

## 10. Fit and Transform

The vectorizer was fitted only on D1-D4, so it learned the vocabulary from the training documents. In N1, `your` and `prize` are out of vocabulary, while in N2, `is`, `moved`, and `tomorrow` are out of vocabulary. `prize` is unknown even though D2 contains `prizes` because they are different tokens, and stemming or lemmatization could help normalize related word forms.

## 11. Data Leakage

When the vectorizer is fitted only on D1-D4, the test-only words are not included in the vocabulary. When N1 and N2 are incorrectly included during fitting, `is`, `moved`, `prize`, `tomorrow`, and `your` enter the vocabulary. This is data leakage because information from the test set has influenced the representation learned from the training data.

## 12. No Meaning

The cosine similarity matrix gives a similarity of approximately `0.75` between every pair of reviews. R1 is therefore not more similar to R2 than to R3 according to the CountVectorizer representation. The reason is that all three reviews share `the`, `food`, and `was`, while the vectorizer treats `tasty`, `delicious`, and `terrible` as unrelated words and does not understand their meanings.

## Vectorizer Choices for a Spam Detector

For a spam detector, I would start with CountVectorizer and compare it with TF-IDF rather than assuming one is always better. I would avoid aggressive English stopword removal because the practical output showed that words such as `now` can be removed, even though they may be useful spam signals, and `not` can also carry important meaning. I would consider unigrams together with selected bigrams because phrases such as `free prize` can provide stronger information than individual words, but I would avoid very large n-gram ranges because the vocabulary grows quickly. I would also use `min_df` or another vocabulary limit only after checking validation performance, because rare words can sometimes be useful indicators of spam.

---

## References

1. Spärck Jones, K. (1972). *A Statistical Interpretation of Term Specificity and Its Application in Retrieval*. Journal of Documentation, 28(1), 11–21.  
   https://doi.org/10.1108/EB026526

2. Robertson, S. (2004). *Understanding Inverse Document Frequency: On Theoretical Arguments for IDF*. Journal of Documentation, 60(5), 503–520.

3. Scikit-learn Developers. *CountVectorizer Documentation*. Scikit-learn.  
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

4. Scikit-learn Developers. *TfidfVectorizer Documentation*. Scikit-learn.  
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

5. Scikit-learn Developers. *Feature Extraction: Text Feature Extraction*. Scikit-learn.  
   https://scikit-learn.org/stable/modules/feature_extraction.html

6. Scikit-learn Developers. *Common Pitfalls and Recommended Practices: Data Leakage*. Scikit-learn.  
   https://scikit-learn.org/stable/common_pitfalls.html

7. Endalie, D., Haile, G., & Taye, W. (2023). *Deep learning-based idiomatic expression recognition for the Amharic language*. PLOS ONE, 18(12), e0295339.  
   https://doi.org/10.1371/journal.pone.0295339

8. Neshir, G., Rauber, A., & Atnafu, S. (2021). *Topic Modeling for Amharic User Generated Texts*. Information, 12(10), 401.  
   https://doi.org/10.3390/info12100401