# Part A -- Research: Text Vectorization and Text Representation in Machine Learning

## Introduction

Natural language is unstructured: people express the same idea using
different words, sentence structures, spelling, and writing styles. Most
traditional machine-learning algorithms, however, require numerical
input. Text vectorization is therefore an important stage in natural
language processing (NLP). It converts documents into numerical features
that an algorithm can process. This section explains common
text-representation methods, their strengths and limitations, and their
use in practical NLP systems.

## 1. What is text vectorization, and why is it necessary in machine learning?

**Text vectorization** is the process of converting text, such as words,
sentences, or complete documents, into numerical representations
(vectors). A machine-learning algorithm generally cannot calculate
directly with a sentence such as "The student studies computer science."
It needs numerical values that represent the text.

A common approach is to first build a **vocabulary**: a collection of
the distinct terms found in the training documents. Each vocabulary term
is assigned a feature position. A document can then be represented by
values at those positions. Depending on the method, the values may
indicate whether a term occurs, how many times it occurs, or how
important it is.

For example, consider the vocabulary:

`[apple, banana, orange]`

A simple one-hot representation of the individual word `banana` is:

`[0, 1, 0]`

The value 1 marks the position assigned to `banana`; the other positions
are 0. For documents, methods such as Bag of Words (BoW) and TF-IDF
create vectors using the same general idea, but the values represent
document-level term information.

**Why it is necessary:**

1.  **Numerical input:** Most traditional machine-learning algorithms
    operate on numbers, not raw text.
2.  **Consistent features:** A vocabulary provides a consistent feature
    space, so documents can be represented using the same vector
    positions.
3.  **Learning patterns:** Algorithms can learn statistical
    relationships between words and labels, such as whether certain
    terms are associated with spam or a particular topic.
4.  **Comparison and retrieval:** Numerical representations make it
    possible to calculate similarities between documents and support
    tasks such as search, classification, and clustering.

**Example:** In a spam-detection system, emails can be converted into
vectors. Terms such as "prize," "winner," and "claim" may become useful
features. A classifier can learn whether their occurrence patterns are
associated with spam.

Vectorization does not automatically understand the meaning of a
sentence. The quality and limitations of the representation depend on
the method used.

## 2. Explain the Bag of Words (BoW) model. What is a Document-Term Matrix (DTM), and how is it used?

The **Bag of Words (BoW)** model represents a document by the words it
contains and, commonly, the frequency of each word. It treats a document
as a collection or "bag" of terms. In its basic form, it does not
preserve grammar or the order in which words appear.

Consider three short documents:

-   Document 1: `the cat eats fish`
-   Document 2: `the dog eats meat`
-   Document 3: `the cat likes meat`

The vocabulary, in a fixed order, is:

`[the, cat, eats, fish, dog, meat, likes]`

The resulting Document-Term Matrix is:

  Document     the   cat   eats   fish   dog   meat   likes
  ---------- ----- ----- ------ ------ ----- ------ -------
  D1             1     1      1      1     0      0       0
  D2             1     0      1      0     1      1       0
  D3             1     1      0      0     0      1       1

A **Document-Term Matrix (DTM)** is a table in which rows represent
documents and columns represent vocabulary terms. Each cell records a
value for a term in a document. In a basic count-based BoW matrix, the
cell contains the number of times that term occurs. In a binary version,
it records only whether the term occurs.

**How it is used:**

-   **Text classification:** DTM rows can be used as input features for
    models that classify documents, such as spam detection or news
    categorization.
-   **Document comparison:** Rows can be compared using measures such as
    cosine similarity.
-   **Clustering:** Documents with similar term distributions can be
    grouped.
-   **Information retrieval:** Term counts can help identify documents
    that contain query terms.

**Advantages:** BoW is straightforward, relatively easy to implement,
and can be effective for tasks where word occurrence is informative.

**Limitations:** It ignores word order and much of the context. For
example, "not good" and "good" contain the same unigram `good`, so a
unigram BoW representation may fail to capture the change in meaning
caused by `not`. It can also produce very large, sparse matrices.

## 3. What is vocabulary size, and why do text datasets often produce sparse matrices?

**Vocabulary size** is the number of distinct terms represented as
features in a text dataset after the selected preprocessing and
vocabulary-building rules have been applied.

For example, in these documents:

-   `cats eat fish`
-   `dogs eat meat`

the vocabulary is `[cats, eat, fish, dogs, meat]`, so its size is 5.

A text matrix is usually **sparse** because each individual document
contains only a small fraction of all the terms in the complete dataset.
Most document-term combinations therefore have a value of zero.

For example, if a dataset has 10,000 documents and a vocabulary of
100,000 terms, each document may contain only a small number of those
terms. Consequently, most of the 10,000 × 100,000 matrix entries are
zero. Scikit-learn's text-feature documentation describes this as a
common property of text data and notes that sparse-matrix formats are
useful for storing it efficiently.

**Why sparsity matters:**

-   Storing every zero wastes memory.
-   Operations on very large dense matrices can be computationally
    expensive.
-   Sparse formats store non-zero values and the information needed to
    locate them, rather than storing every zero.

A common format is **Compressed Sparse Row (CSR)**. It stores non-zero
values, their column indices, and row-pointer information. Libraries
such as SciPy and scikit-learn support sparse representations.

**Ways to control vocabulary size include:**

1.  `min_df`: ignore terms that occur in fewer than a specified number
    or proportion of documents.
2.  `max_df`: ignore terms that occur in more than a specified number or
    proportion of documents, often because they are too common.
3.  `max_features`: limit the vocabulary to a maximum number of selected
    features.
4.  Preprocessing: consistent lowercasing, removal of irrelevant tokens,
    and carefully selected normalization can reduce unnecessary
    variation.

These controls should be selected with the task in mind. Removing a
frequent or rare word is not always beneficial; a term that is common in
one domain may still be important.

## 4. What are n-grams, and how do they improve text representation?

An **n-gram** is a sequence of `n` consecutive tokens. N-grams extend a
unigram representation by allowing a model to use short sequences of
words as features.

The main types are:

-   **Unigram (n = 1):** one token, such as `good`.
-   **Bigram (n = 2):** two consecutive tokens, such as `not good`.
-   **Trigram (n = 3):** three consecutive tokens, such as
    `not very good`.

Consider the sentence:

`The service is not good`

Its unigram features include `the`, `service`, `is`, `not`, and `good`.
Its bigram features include `the service`, `service is`, `is not`, and
`not good`.

**How n-grams improve representation:**

1.  **Some local word order is preserved:** Unlike a unigram BoW model,
    bigrams and trigrams represent short sequences.
2.  **Phrase information is captured:** Expressions such as
    `machine learning`, `credit card`, and `not satisfied` can be
    represented as individual features.
3.  **Potentially better classification:** Phrase features may help
    distinguish categories or sentiment when the relationship between
    neighboring words matters.

For example, a unigram model may give both "good" and "not good" a
feature for `good`. A bigram model can additionally represent
`not good`, helping a sentiment classifier learn that the phrase is
negative.

**Limitations of n-grams:**

-   The vocabulary can grow rapidly as n increases.
-   Many longer sequences occur rarely, creating additional sparsity.
-   More features require more memory and may increase training time.
-   N-grams capture only local sequences; they do not provide a complete
    understanding of long-distance context or meaning.

In scikit-learn's `CountVectorizer` and `TfidfVectorizer`, the
`ngram_range` parameter controls the range. For example, `(1, 2)`
includes unigrams and bigrams.

## 5. What is TF-IDF? Explain how it works, why it is better than simple word counts in some cases, and summarize its historical development.

**TF-IDF** stands for **Term Frequency--Inverse Document Frequency**. It
is a weighting method that estimates how informative a term is for a
document relative to a collection of documents. It combines two ideas:

-   **Term Frequency (TF):** how often a term occurs in a particular
    document.
-   **Inverse Document Frequency (IDF):** how rare or distinctive the
    term is across the document collection.

A commonly presented IDF formula is:

\[ IDF(t) =
`\log`{=tex}`\left`{=tex}(`\frac{N}{df(t)}`{=tex}`\right`{=tex}) \]

where:

-   \(N\) is the total number of documents.
-   (df(t)) is the number of documents containing term (t).

A basic TF-IDF weight is:

\[ TF`\text{-}`{=tex}IDF(t,d) = TF(t,d) `\times `{=tex}IDF(t) \]

Here, (TF(t,d)) is the frequency of term (t) in document (d).

**Example:** Suppose a collection contains 100 documents. The word `the`
occurs in 95 documents, while `cybersecurity` occurs in only 5. The IDF
of `the` will be low because it is very common across the collection.
The IDF of `cybersecurity` will be higher because it is more
distinctive. If both terms occur in one particular document, TF-IDF will
generally give the rarer term a greater weight, assuming comparable term
frequency.

**Why TF-IDF can be better than simple word counts:**

-   A count-based BoW model gives high values to frequent terms even
    when those terms occur in nearly every document.
-   TF-IDF reduces the influence of terms that are common throughout the
    collection.
-   It can emphasize terms that help distinguish one document from
    others.
-   It is often useful for search, document similarity, and traditional
    text-classification tasks.

However, TF-IDF does not understand meaning or context. A rare word is
not necessarily useful, and two words with similar meanings are still
treated as separate features unless additional processing is used.

**Historical development:**

-   **1957 -- Hans Peter Luhn:** Luhn's work on mechanized encoding and
    searching of literary information helped establish statistical
    approaches to selecting and weighting terms for information
    retrieval.
-   **1972 -- Karen Spärck Jones:** Her paper, "A Statistical
    Interpretation of Term Specificity and Its Application in
    Retrieval," developed the inverse-document-frequency idea as a way
    to estimate term specificity.
-   **1988 -- Gerard Salton and Chris Buckley:** Their paper,
    "Term-Weighting Approaches in Automatic Text Retrieval," examined
    and compared term-weighting methods for information retrieval,
    helping establish TF-IDF-style weighting as an important approach.

**Practical implementation:** Scikit-learn's `TfidfVectorizer` combines
count-vectorization and TF-IDF weighting. Its implementation uses
smoothed IDF by default:

\[ IDF(t) =
`\log`{=tex}`\left`{=tex}(`\frac{1+N}{1+df(t)}`{=tex}`\right`{=tex})+1
\]

It also applies L2 normalization by default. These implementation
details differ from the simpler textbook formula, so the exact formula
should be stated when describing a particular implementation.

## 6. Explain the difference between `fit`, `transform`, and `fit_transform`. Why is data leakage a problem?

In scikit-learn, text vectorizers learn information from a collection of
documents and then use that learned information to convert documents
into feature vectors.

-   **`fit()`** learns the representation from the supplied training
    text. For a count vectorizer, this includes building the vocabulary.
    For a TF-IDF vectorizer, it also learns document-frequency
    information used for IDF weighting.
-   **`transform()`** uses the already learned vocabulary and weighting
    information to convert new documents into vectors. It does not learn
    a new vocabulary.
-   **`fit_transform()`** performs both operations: it learns from the
    supplied documents and returns their transformed matrix. It is
    commonly used on the training data.

**Correct train/test procedure:**

1.  Split the dataset into training and test sets.
2.  Fit the vectorizer using only the training text.
3.  Transform the training text.
4.  Transform the test text using the same fitted vectorizer.
5.  Train the model on the training features and evaluate it on the test
    features.

Example:

``` python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

**Data leakage** occurs when information from outside the training data
is used during model training. If a vectorizer is fitted on the complete
dataset before the train/test split, information from the test documents
influences the learned vocabulary and IDF values. The test set is then
no longer fully independent of the training process.

This can produce an overly optimistic evaluation and may make the model
appear to perform better than it would on genuinely unseen data.

The scikit-learn guidance is to split the data first, fit preprocessing
steps only on the training data, and apply the fitted transformations to
the test data. A scikit-learn `Pipeline` can help enforce this procedure
during model selection and cross-validation.

## 7. Case study: Text representation for an African language

### Topic Modeling for Amharic User-Generated Texts

A relevant African-language NLP study is:

Neshir, G., Rauber, A., and Atnafu, S. (2021), "Topic Modeling for
Amharic User Generated Texts," published in *Information*, volume 12,
issue 10, article 401.

The study addresses **Amharic**, an Ethiopian language, and investigates
topic modeling and text classification for user-generated content. The
authors report a dataset of 14,751 documents grouped into ten topic
categories. They examine TF-IDF features, Latent Dirichlet Allocation
(LDA) features, and combinations of these feature types. They also
investigate the effect of stemming and evaluate classifiers including
Support Vector Machines (SVM), Naive Bayes, Logistic Regression, and
Neural Networks.

The paper reports its best stated classification result as **88%
accuracy**, obtained with an SVM using TF-IDF word features and SMOTE,
without stemming. The paper also examines stemming in separate
comparisons; therefore, the result should be understood as the reported
best configuration rather than a general claim that stemming is always
harmful or always unnecessary.

**What the case study demonstrates:**

1.  **TF-IDF can provide useful features:** Statistical word weighting
    can support topic classification in an African language when a
    suitable dataset and preprocessing approach are available.
2.  **Feature choice matters:** The study compares TF-IDF and
    topic-model-based features, showing that representation choices can
    affect classification performance.
3.  **Preprocessing needs evaluation:** Stemming should not
    automatically be assumed to improve results. Its effect depends on
    the language, data, and task.
4.  **Language-specific challenges remain:** Morphology, spelling
    variation, tokenization, available linguistic resources, and limited
    labeled data can make NLP more difficult for languages with fewer
    digital resources.

**Comparison with embeddings:** TF-IDF represents terms as weighted
vocabulary features. It does not directly encode semantic similarity
between different words. Word or sentence embeddings can represent
semantic relationships in a dense numerical space, but they require
suitable pretrained models or sufficient training data. The cited
Amharic study compares TF-IDF and LDA-based feature approaches; it
should not be described as a direct experimental comparison between
TF-IDF and modern neural embeddings.

**Relevance to other African languages:** The study provides a useful
example of how conventional statistical representations can be applied
to an African-language dataset. However, its results cannot simply be
transferred to Somali, Swahili, or another language. Each language
requires appropriate data, preprocessing, tokenization, and evaluation.

## Conclusion

Text vectorization converts unstructured language into numerical
features that machine-learning algorithms can process. Bag of Words
represents term occurrence or frequency, while a Document-Term Matrix
organizes those values across documents. Vocabulary size and sparsity
are important practical concerns because text collections can contain
many unique terms while each document uses only a small subset. N-grams
preserve limited local word order, and TF-IDF gives greater weight to
terms that are distinctive within a document collection. Correct use of
`fit`, `transform`, and `fit_transform` is essential to prevent data
leakage during model evaluation. The Amharic case study illustrates the
practical use of TF-IDF and topic-model features in African-language
NLP, while also showing why representation and preprocessing decisions
should be evaluated for each language and task.

## References

1.  Luhn, H. P. (1957). A statistical approach to mechanized encoding
    and searching of literary information. *IBM Journal of Research and
    Development, 1*(4), 309--317. https://doi.org/10.1147/rd.14.0309

2.  Spärck Jones, K. (1972). A statistical interpretation of term
    specificity and its application in retrieval. *Journal of
    Documentation, 28*(1), 11--21. https://doi.org/10.1108/eb026526

3.  Salton, G., & Buckley, C. (1988). Term-weighting approaches in
    automatic text retrieval. *Information Processing & Management,
    24*(5), 513--523. https://doi.org/10.1016/0306-4573(88)90021-0

4.  Scikit-learn developers. (n.d.). *Feature extraction: Text feature
    extraction*. Scikit-learn documentation.
    https://scikit-learn.org/stable/modules/feature_extraction.html

5.  Scikit-learn developers. (n.d.). *TfidfVectorizer*. Scikit-learn
    documentation.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

6.  Scikit-learn developers. (n.d.). *Common pitfalls and recommended
    practices*. Scikit-learn documentation.
    https://scikit-learn.org/stable/common_pitfalls.html

7.  Neshir, G., Rauber, A., & Atnafu, S. (2021). Topic modeling for
    Amharic user generated texts. *Information, 12*(10), 401.
    https://doi.org/10.3390/info12100401
