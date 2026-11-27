# Assignment 3 – Vectorization

## Part A – Research

### 1. Text vectorization and one-hot encoding

**Text vectorization** is the process of converting words, sentences, or complete documents into numerical representations. Machine-learning algorithms operate on numbers, so vectorization changes language into measurable features that an algorithm can use to identify patterns, compare documents, or make predictions.

Most conventional machine-learning algorithms expect every example to have the same number of input features. A spreadsheet model, for example, expects each row to contain the same columns. Text documents have different lengths and words, so a shared vocabulary is built and each document is represented using that same set of vocabulary positions. A document may have a zero for a word it does not contain, but the feature position still exists.

In **one-hot encoding**, each vocabulary word receives a vector whose length equals the vocabulary size. The position belonging to that word is 1 and every other position is 0. For a vocabulary `[cash, class, free, lunch]`, the vector for `free` is `[0, 0, 1, 0]`. This is useful for representing an individual token, but a sentence contains several tokens. Representing a whole sentence therefore requires several one-hot vectors or an additional rule for combining them. One-hot vectors also do not express relationships: `free` and `cash` are just two different positions, even if they often occur in similar contexts.

**Real-life example:** an SMS spam detector could use a vocabulary containing `free`, `cash`, `win`, and `meeting`. Each message is converted into a fixed-length numerical row. The model can learn that words such as `free` and `win` often occur in promotional messages, while `meeting` may be more common in ordinary communication.

### 2. Bag of Words and the document-term matrix

The **Bag of Words (BoW)** method represents a document by counting how many times each vocabulary word occurs. First, the system builds one vocabulary from the collection. It then creates a **document-term matrix (DTM)**: rows represent documents, columns represent vocabulary terms, and each cell records the count of that term in that document. A binary version records only whether a word occurs; a count version records its frequency.

It is called a “bag” because it treats a document as if its words were placed in a bag: the method keeps the words and their counts but normally discards their original order. For example, `win free phone` and `phone free win` receive the same BoW vector.

Losing word order can be acceptable for broad topic classification, such as identifying whether a long article is mainly about sports or agriculture, because the topic words may be informative even without their exact positions. It can cause serious problems for sentiment or intent analysis. The sentences `the service was good` and `the service was not good` share most of their words, but the second expresses the opposite opinion. A unigram BoW representation may not preserve the effect of `not`.

### 3. Vocabulary size and sparsity

The **vocabulary** is the set of distinct tokens selected from the training corpus. Its size is the number of features (columns) in the DTM. A large collection can contain many thousands or millions of distinct terms. Documents usually contain only a small fraction of those terms, so most cells in the DTM are zero. This property is called **sparsity**.

Sparse matrices are stored efficiently by recording non-zero values and their positions rather than storing every zero. Common sparse formats include Compressed Sparse Row (CSR), which is efficient for row operations, and Compressed Sparse Column (CSC), which is useful for column operations. Scikit-learn vectorizers return sparse matrices by default to avoid wasting memory.

Vocabulary controls affect the features retained:

- `min_df` removes terms that occur in fewer than the specified number or proportion of documents. It can remove misspellings or rare noise, but may also remove a rare yet useful spam indicator.
- `max_df` removes terms occurring in more than the specified number or proportion of documents. It can filter extremely common terms that do not distinguish documents.
- `max_features` keeps only the most frequent terms up to a specified limit. This reduces memory and training cost but may discard less frequent informative terms.
- Lowercasing merges case variants such as `Win` and `WIN` into `win`.
- Stopword removal removes selected common words. This can reduce the vocabulary, but removing negation words such as `not` can damage sentiment information.
- Stemming reduces related word forms to a common stem. For example, a stemmer may reduce inflected forms to a shared base-like form. This can reduce vocabulary size, although stems are not always real dictionary words and aggressive stemming can merge words that should remain distinct.

Preprocessing should be chosen for the language and task, not applied automatically. In morphologically rich or low-resource languages, a generic English stopword list or stemmer may be inappropriate.

### 4. N-grams

An **n-gram** is a consecutive sequence of *n* tokens. A unigram contains one token, a bigram contains two consecutive tokens, and a trigram contains three. For the sentence `the service was not good`, unigrams include `not` and `good`; bigrams include `not good`; and trigrams include `was not good`.

N-grams preserve limited local word order. The bigram `not good` is different from the unigram features `not` and `good` considered separately, and it can help a classifier learn the phrase's negative meaning. Other useful phrases include `limited offer` and `click here` in spam detection.

Larger n-gram ranges increase the number of possible features. Many longer sequences occur only once, which creates a larger and sparser matrix and can increase memory use and overfitting. A practical system normally tests a small range, such as unigrams plus bigrams, rather than assuming that longer sequences are always better.

### 5. TF-IDF and inverse document frequency

**Term frequency–inverse document frequency (TF-IDF)** is a weighting method used in information retrieval and text mining. It increases the weight of a term when it occurs frequently in a particular document, while reducing the weight of terms that occur in many documents. This helps distinguish a document's characteristic words from words shared by almost every document.

The inverse-document-frequency idea developed within automatic indexing and information retrieval research. Early work by Karen Spärck Jones examined statistical interpretation of term specificity in 1972. Salton, Wong, and Yang's vector-space-model work in 1975 helped establish term weighting within automatic document indexing and retrieval.

A common textbook definition is:

\[
\operatorname{TF}(t,d)=\text{number of occurrences of }t\text{ in }d
\]

\[
\operatorname{IDF}(t)=\log\left(\frac{N}{df(t)}\right)
\]

\[
\operatorname{TFIDF}(t,d)=\operatorname{TF}(t,d)\times\operatorname{IDF}(t)
\]

Here, \(N\) is the number of documents and \(df(t)\) is the number of documents containing term \(t\). If a word appears in almost every document, its IDF is small. If it appears in only a few documents, its IDF is larger, because it is more useful for distinguishing documents.

Scikit-learn's `TfidfVectorizer` is not exactly the unsmoothed textbook calculation. By default, it uses smoothed IDF:

\[
\operatorname{IDF}_{sklearn}(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

It adds one to the numerator and denominator to smooth the estimate and adds one to the resulting IDF. It also applies L2 normalization to each document vector by default, so each non-zero row has Euclidean length 1. Its default term-frequency component is the raw count unless options such as `binary=True` or `sublinear_tf=True` are selected.

TF-IDF values are **weights, not probabilities**. They are not required to lie between zero and one before normalization, do not represent the chance of a word occurring, and do not sum to one as a probability distribution. L2 normalization makes the vector length one, but that is a geometric scaling operation, not probability calibration.

### 6. `fit`, `transform`, `fit_transform`, and data leakage

`fit` learns the vocabulary and feature indices from the supplied training documents. For TF-IDF, it also learns document frequencies and IDF weights. `transform` uses the already learned vocabulary and weights to convert new documents into vectors. `fit_transform` performs both operations on the same input and is a convenient method for the training set.

A word not present in the fitted vocabulary is an **out-of-vocabulary (OOV)** word. A standard scikit-learn vectorizer ignores it during transformation, so it contributes no feature to the resulting row. For example, if the training vocabulary contains `prizes` but not `prize`, the two forms are different tokens unless preprocessing such as stemming or lemmatization maps them to a common form.

**Data leakage** happens when information from the test set is used while learning preprocessing or model parameters. If a vectorizer is fitted on both training and test documents, the vocabulary and IDF statistics have been influenced by the test data. This gives the training process information it would not have in a real deployment and can make evaluation optimistic or otherwise unrepresentative.

The correct order is to split the data first, fit preprocessing and vectorization on the training text only, transform the training text, transform the test text using that same fitted vectorizer, train the model on training vectors, and evaluate on test vectors. A scikit-learn `Pipeline` helps keep these steps together and reduces accidental leakage during cross-validation.

### 7. Case study: text classification in an African language

A relevant African-language example is the 2026 study by Demilie and Zia, *News Classification in Low-Resource Languages: Insights From Transformer and Baseline Models*, which investigates Amharic news classification. The study compares traditional feature-based approaches, including TF-IDF with logistic regression and Word2Vec with a multilayer perceptron, with pretrained African-language transformer models AfriBERTa and AfroXLMR. The reported macro-F1 scores include 90.34% for TF-IDF plus logistic regression, 88.17% for Word2Vec plus an MLP, 92.42% for AfroXLMR, and 94.12% for AfriBERTa.

The comparison shows that TF-IDF can be a strong and interpretable baseline, but a representation based mainly on individual term weights does not directly encode semantic similarity. Word2Vec represents words through learned contextual/distributional vectors, while transformer models build context-sensitive representations and can use surrounding words to help interpret a phrase. The better reported AfriBERTa result suggests the value of language-adapted contextual representations for this task. It does not prove that transformers always win: dataset quality, model size, training setup, class balance, and evaluation design all affect results. The study is useful because it illustrates the trade-off between efficient lexical baselines and richer representations in a low-resource African language.

A second relevant example is Tunga and David's 2025 Swahili social-media sentiment study. It used 3,000 Zindi Africa tweets, TF-IDF features, and five classical machine-learning algorithms. CatBoost achieved the strongest reported result among those models, with 0.610 accuracy and 0.470 F1. The modest F1 score highlights the difficulty of sentiment analysis in a low-resource language and the limits of relying on lexical frequency features alone.


## References

1. Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM, 18*(11), 613–620. https://doi.org/10.1145/361219.361220
2. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526
3. scikit-learn developers. (2026). *TfidfVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
4. scikit-learn developers. (2026). *Feature extraction: Text feature extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html
5. Demilie, W. W. B., & Zia, A. (2026). News classification in low-resource languages: Insights from transformer and baseline models. *Concurrency and Computation: Practice and Experience*. https://doi.org/10.1002/cpe.70648
6. Tunga, M., & David, D. (2025). A comparative study for classical machine learning models for Swahili social media sentiment analysis. *Tanzania Journal of Science, 51*(3). https://doi.org/10.65085/2507-7961.1051
7. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
