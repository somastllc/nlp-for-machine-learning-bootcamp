# Assignment 3 – Vectorization

**Student:** Hamza-Idiris  
**Submission folder:** `submissions/Hamza-Idiris/assignment-03/`

---

# Part A – Research

## 1. Text vectorization and one-hot encoding

Text vectorization is the process of converting words or documents into numerical features that a machine-learning algorithm can process. Most traditional machine-learning algorithms expect a fixed-length numerical input, while natural-language text can contain different words and different numbers of words. A vectorizer therefore creates a vocabulary and assigns each vocabulary item a feature position. Every document is represented using the same feature positions, even when some values are zero.

In one-hot encoding, each vocabulary word receives a vector containing zeros everywhere except a one at that word's position. For example, with the vocabulary `[cash, class, free, lunch, notes, win]`, the word `free` becomes `[0, 0, 1, 0, 0, 0]`. One-hot encoding is useful for representing individual categories or tokens, but it does not by itself represent a whole document in a compact way. A sentence needs a sequence of token vectors, and the representation does not express similarity between related words. For example, a simple email filter could represent the words in an email, but one-hot vectors for `free` and `prize` do not indicate that both may be associated with promotional messages.

## 2. Bag of Words and Document-Term Matrix

Bag of Words (BoW) represents a document by counting how often each vocabulary word occurs. The word “bag” means that the model treats the document like an unordered collection of words: it keeps word frequency but ignores the original word order and most grammar. A Document-Term Matrix (DTM) is the table produced from a collection of documents. Each row represents one document, each column represents one vocabulary term, and each cell contains that term's count in that document.

Ignoring word order can be acceptable for a basic topic-classification task, such as identifying whether a news article is about sports based on terms such as `team`, `goal`, and `player`. It can be harmful when word order changes meaning or sentiment. For example, “not good” and “good” contain the same unigram `good`, but the word `not` reverses the opinion. A unigram BoW model does not fully capture that relationship.

## 3. Vocabulary size and sparsity

Vocabulary size is the number of distinct features learned from the training corpus. It can become large because documents contain many different words, names, spellings, and inflected forms. A DTM is usually sparse because an individual document uses only a small fraction of all words in the complete corpus. Therefore, most document-term cells contain zero.

Sparse matrices store non-zero values and their positions rather than storing every zero. This saves memory and can make computation more efficient. In scikit-learn, `CountVectorizer` returns a sparse matrix by default. The `min_df` parameter removes terms that occur in fewer than a specified number or proportion of documents; `max_df` removes terms that occur in too many documents; and `max_features` limits the vocabulary to the most frequent features. Lowercasing merges forms such as `Win` and `WIN`; stop-word removal removes selected common words; and stemming can merge related word forms such as `prize` and `prizes`. These steps can reduce vocabulary size, but they must be selected carefully because useful meaning can also be removed.

## 4. N-grams

An n-gram is a sequence of n consecutive tokens. A unigram contains one word, a bigram contains two consecutive words, and a trigram contains three consecutive words. Unigrams represent individual terms, while bigrams and trigrams preserve some local word order. For example, the bigram `not good` preserves a negative phrase that separate unigram features may not represent clearly.

Larger n-gram ranges can improve a model's ability to recognize phrases, but they also increase the number of possible features. Many longer phrases may occur only once, which can make the feature matrix larger and sparser. Longer n-grams can also be less useful when the training corpus is small because the model has fewer examples of each phrase.

## 5. TF-IDF

TF-IDF means Term Frequency–Inverse Document Frequency. It gives a term a weight based on how often it appears in a document and how rare it is across the document collection. Term frequency (TF) measures a term's occurrence in a document. In a simple textbook version, inverse document frequency is:

\[
IDF(t) = \log\left(\frac{N}{df(t)}\right)
\]

where \(N\) is the total number of documents and \(df(t)\) is the number of documents containing term \(t\). The weight is:

\[
TFIDF(t,d) = TF(t,d)\times IDF(t)
\]

A term appearing in many documents receives a lower IDF because it is less useful for distinguishing one document from another. A rarer term receives a higher IDF. Salton and Buckley's 1988 work is an important reference on term-weighting approaches in automatic text retrieval.

Scikit-learn's `TfidfVectorizer` has defaults that differ from the simple formula. By default, it smooths IDF using:

\[
IDF_{\text{sklearn}}(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

It also uses L2 normalization by default, so each document vector is scaled to unit length. Its default term frequency is the raw count unless options such as `sublinear_tf=True` are selected. TF-IDF values are feature weights, not probabilities; they do not have to add up to one.

## 6. fit, transform, fit_transform, and data leakage

`fit()` learns information from the supplied training documents, such as the vocabulary and document frequencies. `transform()` applies the learned vocabulary and settings to new documents and returns their numerical representation. `fit_transform()` performs both operations on the same supplied data and is commonly used for training data.

An out-of-vocabulary (OOV) word is a word that was not included in the vocabulary learned during fitting. With the default `CountVectorizer`, an unseen word does not receive a new feature during `transform()`; it is ignored. Data leakage happens when information from the test set is used while preparing or training the model. For example, fitting a vectorizer on both training and test documents lets the vocabulary learn from test data. The correct order is to split the data first, fit preprocessing and vectorization only on training data, then transform the test data using the already-fitted objects. In a real machine-learning workflow, a scikit-learn `Pipeline` can help keep these steps together.

## 7. Case study: text classification for Tigrinya

Fesseha and colleagues studied text classification for Tigrinya, a low-resource language. Their work used a manually annotated dataset of 30,000 news texts in six categories: sport, agriculture, politics, religion, education, and health. They also described an unannotated corpus containing more than six million words. The researchers compared word representation approaches including Word2Vec CBOW, Word2Vec Skip-gram, and FastText in CNN-based classification, and compared their results with traditional machine-learning approaches.

The paper reports that the CNN using Word2Vec CBOW achieved the best accuracy, 93.41%, for the reported Tigrinya news classification experiment. This case study shows why representation matters for languages with limited resources and complex morphology. BoW and TF-IDF are useful because they are relatively simple and interpretable, but they mainly record token occurrence and weighting. They do not inherently understand synonyms, word meaning, or context. For example, a count-based model may treat `tasty` and `delicious` as unrelated words even though they have similar meanings. Word embeddings can represent words using learned numerical relationships, although their quality depends on available data and training.

### Part A references

1. Fesseha, A. et al. (2021). “Text Classification Based on Convolutional Neural Networks and Word Embedding for Low-Resource Languages: Tigrinya.” *Information*, 12(2), 52. https://doi.org/10.3390/info12020052
2. Salton, G., & Buckley, C. (1988). “Term-weighting approaches in automatic text retrieval.” *Information Processing & Management*, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
3. scikit-learn documentation. “Feature extraction.” https://scikit-learn.org/stable/modules/feature_extraction.html
4. scikit-learn documentation. “CountVectorizer.” https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
5. scikit-learn documentation. “TfidfVectorizer.” https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
6. scikit-learn documentation. “TfidfTransformer.” https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

---

# Part B – Practical

The practical work is implemented in `vectorization_practice.py`. The complete program output is saved in `output.txt` after running the script. The tables and observations below are based on that run.

## Task 1 – Custom one-hot encoding

The vocabulary order is `[cash, class, free, lunch, notes, win]`. The vector for `free` is `[0, 0, 1, 0, 0, 0]`, and the vector for `cash` is `[1, 0, 0, 0, 0, 0]`. The sentence `free cash` contains two word vectors, while `win free cash` contains three word vectors. Therefore, the number of token vectors depends on the number of words in the sentence, while each individual word vector has the same fixed length as the vocabulary.

## Task 2 – CountVectorizer and DTM

The default `CountVectorizer` learned 32 features from the five training documents. Each row below is a document and each column is a vocabulary word. The value in a cell is the number of times that word appears in that document.

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |


D1 contains the word `win` twice, so its `win` cell has value 2. The transformed queries `win free phone` and `phone free win` have identical count vectors because BoW counts the same words without preserving their order. This demonstrates both the usefulness and the limitation of a basic DTM.

## Task 3 – Token count and vocabulary

There are 42 word occurrences in the five documents when counting one-character words as well. The default `CountVectorizer` vocabulary has 32 features because its default token pattern ignores one-character tokens such as `a`. The words that appear in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win`.

## Task 4 – Lowercasing and stop words

With the default settings, `Win`, `FREE`, and `WIN` are lowercased and combined into the feature `win`. With `lowercase=False`, the vocabulary keeps `FREE`, `WIN`, and `Win` as separate features. Punctuation such as `!` is not a word feature, and the one-character token `a` is excluded by the default token pattern. English stop-word filtering removes `to` and `now` from D1; it also removes `not` from the five-document vocabulary. This is important because `not` changes the meaning of D5, so blindly removing all standard English stop words can damage sentiment information.

## Task 5 – Vocabulary limits

The full vocabulary contains 32 features. Using `min_df=2` and `max_features=5` reduces it to five features: `free`, `lecture`, `now`, `the`, and `win`. The `min_df` setting first excludes terms that occur in only one document, and `max_features` then limits the retained vocabulary to five features. This reduces dimensionality, although a rare but useful word could be removed.

## Task 6 – N-grams

For D5, unigrams include `not` and `good` separately, while the bigram vocabulary includes `not good`. That bigram preserves a useful negative opinion phrase. Across all five documents, the vocabulary grows from 32 features with unigrams to 67 with unigrams and bigrams, and to 98 with unigrams, bigrams, and trigrams. The increase shows that n-grams preserve more local context but also increase feature count and sparsity.

## Task 7 – TF-IDF

The TF-IDF matrix is printed in `output.txt` rounded to two decimal places. The following table lists every term present in D2, its raw count, and its scikit-learn TF-IDF weight.

| Word | Count in D2 | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.3891 |
| cash | 1 | 0.3891 |
| entry | 1 | 0.3891 |
| free | 1 | 0.3140 |
| now | 1 | 0.3140 |
| prizes | 1 | 0.3891 |
| today | 1 | 0.3140 |
| win | 1 | 0.3140 |


All terms in D2 occur once, but their TF-IDF weights are not all equal. Terms such as `cash`, `call`, `entry`, and `prizes` occur in only one of the five documents and therefore receive higher IDF values. Terms such as `free`, `now`, `today`, and `win` occur in more than one document and receive lower IDF values. In D1, `win` occurs twice and receives the largest D1 weight in this example because its count is doubled and it is not present in the other documents.

## Task 8 – Manual TF-IDF calculation

For D2, the corpus contains \(N=5\) documents. The term `cash` occurs once in D2 and in one document overall, so \(TF=1\), \(df=1\), and \(IDF=\ln(5/1)=1.609438\). Its manual TF-IDF is \(1\times1.609438=1.609438\).

The term `today` occurs once in D2 and appears in two documents overall, so \(TF=1\), \(df=2\), and \(IDF=\ln(5/2)=0.916291\). Its manual TF-IDF is \(1\times0.916291=0.916291\). The manual ranking gives `cash` a higher weight than `today`, which agrees with the relative IDF ranking in scikit-learn. The numerical values differ because the manual calculation uses unsmoothed IDF and raw counts, while scikit-learn smooths IDF and applies L2 normalization by default.

## Task 9 – Sparse matrix statistics

The DTM has shape \(5\times32\), giving 160 total cells. There are 39 non-zero cells and 121 zero cells. Therefore, \(121/160\times100=75.62\%\) of the matrix cells are zero. This confirms that the document-term matrix is sparse.

## Task 10 – Out-of-vocabulary words

The vectorizer is fitted only on D1–D4 and then used to transform N1 and N2. In N1, `prize` and `your` are out of vocabulary. In N2, `is`, `moved`, and `tomorrow` are out of vocabulary. `prize` is not the same token as `prizes`, even though `prizes` occurs in D2. A stemming step such as Porter stemming can reduce both `prize` and `prizes` to the same stem, allowing them to share a feature if the same stemming procedure is applied consistently before vectorization.

## Task 11 – Data leakage

When the vectorizer is fitted on D1–D4 only, the test-only words do not appear in its vocabulary. If it is incorrectly fitted on D1–D4 plus N1 and N2, the new vocabulary includes `is`, `moved`, `prize`, `tomorrow`, and `your`. These words have come from the test documents, demonstrating how fitting preprocessing on test data leaks information about the test set. The correct approach is to fit the vectorizer on training documents only and use `transform()` for test documents.

## Task 12 – Cosine similarity

The review vocabulary is `delicious`, `food`, `tasty`, `terrible`, `the`, and `was`. The cosine similarity between R1 (`the food was tasty`) and R2 (`the food was delicious`) is 0.750. The similarity between R1 and R3 (`the food was terrible`) is also 0.750. Therefore, the count-based representation does not consider the tasty review more similar to delicious than to terrible. It sees the three shared words and does not understand that `tasty` and `delicious` are synonyms or that `terrible` has an opposite sentiment.

## Spam-detector recommendation

For a basic spam detector using this small example, I would begin with TF-IDF features and compare them against raw counts using the same train/test split and evaluation metrics. I would keep negation words such as `not` rather than removing all English stop words, because they can change the meaning of a message. I would test unigrams together with bigrams, since phrases such as `not good` or `free entry` may be more informative than isolated words, but I would avoid large n-gram ranges at first because the vocabulary grows quickly. I would use `min_df` and `max_features` only after checking validation performance, so that rare but important spam terms are not removed without evidence. The practical results show why these choices matter: the full vocabulary has 32 features, the limited vocabulary has only five, and adding bigrams and trigrams increases the feature count to 67 and 98.

---

## Files included

- `answers.md` – Part A research and Part B explanations, including the required DTM and D2 TF-IDF tables.
- `vectorization_practice.py` – complete practical implementation for all 12 tasks.
- `output.txt` – complete output captured by running the Python script.
