# Assignment 3 – Vectorization

## Part A – Research

**Goal:** Understand how text is converted into numerical representations that Machine Learning models can process, including One-Hot Encoding, Bag of Words, Document-Term Matrices, vocabulary, sparsity, n-grams, TF-IDF, fitting and transforming, data leakage, and the limitations of traditional text representations.

---

## Question 1: Text Vectorization and One-Hot Encoding

### 1.1 Definition of Text Vectorization

Text vectorization is the process of converting written text into numerical representations that computers and Machine Learning algorithms can understand and process.

Computers cannot directly perform mathematical calculations on ordinary sentences in the same way they process numerical data. Therefore, text must first be transformed into numerical features.

Common text vectorization techniques include:

- One-Hot Encoding
- Bag of Words (BoW)
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Word embeddings

Each technique represents text differently and preserves different types of information.

### 1.2 Why Do Machine Learning Models Need the Same Number of Features?

Most traditional Machine Learning algorithms require every input document to have the same number of features.

A feature represents one measurable property of the input data. When documents are converted into vectors, every position must represent the same feature across all documents.

For example, consider a vocabulary containing five words:

`["computer", "education", "money", "school", "work"]`

Every document must be represented using these five positions, even if some words do not appear in a particular document.

A missing word receives a value of zero.

This consistent structure allows Machine Learning algorithms to compare documents, calculate distances, and perform mathematical operations.

### 1.3 One-Hot Encoding

One-Hot Encoding is a technique that represents each word using a binary vector.

The vector contains:
- `1` in the position assigned to the word.
- `0` in all other positions.

For example:

Vocabulary:

`["apple", "book", "computer", "school"]`

| Word | One-Hot Vector |
|---|---|
| apple | [1, 0, 0, 0] |
| book | [0, 1, 0, 0] |
| computer | [0, 0, 1, 0] |
| school | [0, 0, 0, 1] |

Each word has a unique numerical representation.

However, One-Hot Encoding does not show semantic relationships between words. For example, the vectors for "apple" and "orange" would be equally different from each other as the vectors for "apple" and "computer".

### 1.4 Why Is One-Hot Encoding Not Enough for a Whole Document?

One-Hot Encoding represents individual words rather than the complete meaning of a document.

A document can contain many words, and representing all those words requires combining their individual representations.

One-Hot Encoding also has important limitations:

1. It creates large vectors when the vocabulary is large.
2. It does not capture semantic similarity.
3. It does not naturally represent relationships between words.
4. It does not independently provide a convenient fixed-length representation of an entire document.

Other techniques, such as Bag of Words and TF-IDF, are commonly used to represent complete documents.

### 1.5 Real-Life Example

Consider an email classification system that separates promotional emails from ordinary messages.

Email A:

"Congratulations! You have received a special offer."

Email B:

"The school meeting will take place tomorrow."

One-Hot Encoding can represent individual words such as "offer", "school", and "meeting" using numerical vectors.

However, it cannot independently understand the complete message. A document-level representation such as Bag of Words or TF-IDF can convert each email into a fixed-length numerical vector that a classification algorithm can use.

**Conclusion:** Text vectorization converts written language into numerical features. One-Hot Encoding provides a basic way to represent individual words, but more advanced methods are needed to represent complete documents effectively.

---

## Question 2: Bag of Words and the Document-Term Matrix

### 2.1 What Is Bag of Words?

Bag of Words (BoW) is a text representation technique that converts documents into numerical vectors based on the words they contain.

It creates a vocabulary from a collection of documents and counts how many times each vocabulary word appears in each document.

Bag of Words focuses on word occurrence and frequency rather than grammar or sentence structure.

### 2.2 What Is a Document-Term Matrix?

A Document-Term Matrix (DTM) is a table that represents the relationship between documents and vocabulary terms.

It contains three main components:

- **Rows:** Represent individual documents.
- **Columns:** Represent unique words or features in the vocabulary.
- **Cells:** Represent the number of times a word appears in a document.

Example:

Suppose we have three documents:

- D1: "I love machine learning"
- D2: "I love programming"
- D3: "Machine learning is useful"

Vocabulary:

`["i", "love", "machine", "learning", "programming", "is", "useful"]`

Document-Term Matrix:

| Document | i | love | machine | learning | programming | is | useful |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| D2 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| D3 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |

The matrix shows how frequently each vocabulary word occurs in every document.

### 2.3 Why Is It Called a "Bag"?

It is called a Bag of Words because the method treats a document like a collection or bag of words.

The original order of words is not preserved. The method mainly focuses on which words appear and how frequently they occur.

For example:

- Sentence A: "The student likes programming."
- Sentence B: "Programming likes the student."

Both sentences contain the same words. A basic Bag of Words representation produces the same word-count vector for both sentences.

### 2.4 When Does Losing Word Order Cause Little Harm?

Losing word order can cause little harm in some document classification tasks.

For example, identifying whether a document is related to sports, technology, or education can often be done using important keywords.

Words such as "football", "player", and "match" may help identify a sports-related document even without their original order.

### 2.5 When Does Losing Word Order Cause Problems?

Losing word order can cause serious problems in sentiment analysis and other tasks that depend on sentence meaning.

Consider:

- "The movie is good."
- "The movie is not good."

The word "not" changes the meaning and sentiment of the sentence.

A simple Bag of Words model may recognize the same main words but cannot fully understand how their arrangement changes the meaning.

**Conclusion:** Bag of Words is a simple and useful technique for converting text into numerical features, but it loses word order and important contextual information.

---

## Question 3: Vocabulary Size and Sparsity

### 3.1 What Is Vocabulary Size?

Vocabulary size refers to the total number of unique words or features identified in a collection of documents.

For example, consider:

- D1: "I study computer science"
- D2: "I study machine learning"

The vocabulary is:

`["i", "study", "computer", "science", "machine", "learning"]`

The vocabulary size is six because there are six unique words.

Vocabulary size directly affects the number of features in a Document-Term Matrix.

### 3.2 What Is Sparsity?

Sparsity refers to a situation where most values in a matrix are zero.

In text data, individual documents normally contain only a small portion of all the words in the complete vocabulary.

Therefore, most document-term matrix cells contain zero.

For example, a document containing 20 words may be represented using a vocabulary containing 10,000 features. Only a small number of those features will have non-zero values.

### 3.3 Why Do Real Document-Term Matrices Become Large and Sparse?

Real document collections can contain thousands or millions of documents.

Different documents use different words, and the vocabulary grows as new words appear.

This creates a matrix with:

- Many rows representing documents.
- Many columns representing unique terms.
- A large number of zero values because most documents do not contain most vocabulary words.

A large document-term matrix can therefore require significant memory if every zero is stored individually.

### 3.4 How Are Sparse Matrices Stored Efficiently?

Sparse matrices are designed to store non-zero values and their positions rather than storing every zero.

For example, instead of storing:

`[0, 0, 3, 0, 0, 1, 0]`

A sparse representation stores the non-zero values and the positions where they occur.

Common sparse matrix formats include:

- **CSR (Compressed Sparse Row):** Efficient for row-based operations and matrix calculations.
- **CSC (Compressed Sparse Column):** Efficient for column-based operations.
- **COO (Coordinate Format):** Stores row indices, column indices, and non-zero values.

Scikit-learn commonly uses sparse matrices for text vectorization because they reduce memory usage.

### 3.5 How Do Vectorizer Settings Change the Vocabulary?

Several parameters can control the size of the vocabulary.

| Parameter | Meaning | Effect |
|---|---|---|
| `min_df` | Minimum document frequency | Removes words that appear in too few documents. |
| `max_df` | Maximum document frequency | Removes words that appear in too many documents. |
| `max_features` | Maximum vocabulary size | Keeps only a limited number of features, generally selected by term frequency. |

For example:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    min_df=2,
    max_df=0.90,
    max_features=1000
)
```

This configuration:
- Removes terms appearing in fewer than two documents.
- Removes terms appearing in more than 90% of documents.
- Limits the vocabulary to at most 1,000 features.

### 3.6 How Does Preprocessing Change the Vocabulary?

Preprocessing also affects vocabulary size.

**Lowercasing**

Converts uppercase letters into lowercase letters.

For example:

`"Machine"` and `"machine"` become the same token.

This reduces duplicate vocabulary entries caused by capitalization.

**Stopword Removal**

Removes frequently occurring words that may contribute little useful information to a particular task.

Examples include "the", "is", and "and".

However, removing stopwords must be done carefully because words such as "not" can change sentence meaning.

**Stemming**

Reduces related word forms to a common stem.

For example, "playing", "played", and "plays" may be reduced to "play".

This can reduce vocabulary size, although stemming sometimes produces word forms that are not actual dictionary words.

**Conclusion:** Vocabulary size determines the number of text features, while sparsity describes how many matrix values are zero. Sparse storage and vocabulary filtering help make text processing more efficient.

---

## Question 4: N-Grams

### 4.1 What Are N-Grams?

N-grams are continuous sequences of words or characters taken from a text.

They help preserve some information about word order that a basic Bag of Words model loses.

The letter `n` represents the number of consecutive units in a sequence.

The three common types are unigrams, bigrams, and trigrams.

### 4.2 Unigrams

A unigram contains one word at a time.

Example:

Sentence:

"The student studies machine learning."

Unigrams:

- The
- student
- studies
- machine
- learning

Unigrams represent individual words but do not preserve relationships between neighboring words.

### 4.3 Bigrams

A bigram contains two consecutive words.

Example:

Sentence:

"The student studies machine learning."

Bigrams:

- The student
- student studies
- studies machine
- machine learning

Bigrams preserve some word order by representing pairs of neighboring words.

### 4.4 Trigrams

A trigram contains three consecutive words.

Example:

Sentence:

"The student studies machine learning."

Trigrams:

- The student studies
- student studies machine
- studies machine learning

Trigrams preserve more local context than unigrams and bigrams.

### 4.5 Example Where a Bigram Changes Meaning

Consider these two expressions:

- "not useful"
- "very useful"

The unigram representation identifies the words "not", "very", and "useful" separately.

A bigram representation can identify the phrases "not useful" and "very useful", preserving information about how the words are connected.

This is especially useful in sentiment analysis because words such as "not" can reverse the meaning of another word.

### 4.6 The Cost of Using Larger N-Grams

Larger n-grams provide more information about local word order, but they also introduce disadvantages:

1. They increase vocabulary size.
2. They create more features.
3. They increase memory and computational requirements.
4. Many longer sequences appear only once, creating rare features.
5. They can cause overfitting when the training dataset is small.

For example, a trigram such as "the student studies" may appear in only one document, making it less useful for generalizing to unseen documents.

**Conclusion:** N-grams preserve some local word order and can improve text classification, but larger n-grams increase feature space and computational costs.

---

## Question 5: TF-IDF

### 5.1 What Is TF-IDF?

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It is a numerical weighting technique used to estimate how important a word is within a particular document compared with a complete collection of documents.

TF-IDF combines two measurements:

- **Term Frequency (TF):** Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** Measures how uncommon a term is across the document collection.

A word that appears frequently in one document but rarely in other documents may receive a high TF-IDF weight.

### 5.2 Where Did Inverse Document Frequency Come From?

The statistical idea behind inverse document frequency is strongly associated with the work of Karen Spärck Jones.

In 1972, she published a research paper titled:

"A Statistical Interpretation of Term Specificity and Its Application in Retrieval."

Her research explained that terms appearing in fewer documents can be more useful for distinguishing relevant documents than terms appearing in almost every document.

This idea became an important foundation of information retrieval and modern TF-IDF weighting.

### 5.3 Textbook TF-IDF Formula

The traditional formula is:

**Term Frequency:**

\[
TF(t,d)=f_{t,d}
\]

Where:

- \(t\) represents a term.
- \(d\) represents a document.
- \(f_{t,d}\) is the number of times the term appears in the document.

**Inverse Document Frequency:**

\[
IDF(t)=\log\left(\frac{N}{DF(t)}\right)
\]

Where:

- \(N\) is the total number of documents.
- \(DF(t)\) is the number of documents containing the term.

**TF-IDF:**

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

These are common textbook definitions. Other TF and IDF variants also exist.

### 5.4 Why Do Rarer Words Receive Higher Weights?

A word that appears in almost every document does not help distinguish one document from another very well.

For example, words such as "the" and "is" may occur in many documents.

However, a word such as "neuroscience" may appear in only a small number of documents.

Because the rarer word is more useful for distinguishing documents, it receives a higher IDF value.

For example, suppose a collection contains 100 documents:

- The word "computer" appears in 50 documents.
- The word "neuroscience" appears in 5 documents.

Their IDF values are:

\[
IDF(computer)=\log(100/50)
\]

\[
IDF(neuroscience)=\log(100/5)
\]

The second value is higher because "neuroscience" appears in fewer documents.

### 5.5 How Does Scikit-Learn's TfidfVectorizer Differ?

Scikit-learn's `TfidfVectorizer` includes additional options and uses a slightly different default formula from the simplest textbook version.

Its default smoothed IDF formula is:

\[
IDF(t)=\log\left(\frac{1+N}{1+DF(t)}\right)+1
\]

The main differences include:

**1. Smoothing**

Scikit-learn adds one to the total document count and document frequency.

This prevents division by zero and provides a more stable calculation.

**2. Normalization**

By default, `TfidfVectorizer` uses L2 normalization.

This makes each non-zero document vector have a Euclidean length of one.

**3. Term Frequency**

By default, it uses raw term counts. When `sublinear_tf=True`, it applies logarithmic term-frequency scaling.

Example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning is useful",
    "machine learning is powerful",
    "computer science is interesting"
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
```

### 5.6 Why Are TF-IDF Values Weights, Not Probabilities?

TF-IDF values are weights that represent the relative importance of terms in documents.

They are not probabilities because they are not required to:

- Fall between zero and one.
- Add up to one across all terms.
- Represent the likelihood of an event.

Normalization may change the scale of the weights, but it does not turn them into probabilities.

**Conclusion:** TF-IDF gives more importance to terms that are frequent in a document but uncommon across the overall collection. Scikit-learn adds smoothing and normalization to make the representation more practical.

---

## Question 6: Fit, Transform, Out-of-Vocabulary Words, and Data Leakage

### 6.1 What Does Fit Mean?

The `fit()` method learns the necessary information from training documents.

For example, when using `CountVectorizer`, fitting identifies the vocabulary and assigns a numerical feature index to each term.

Example:

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "machine learning",
    "computer science"
]

vectorizer = CountVectorizer()

vectorizer.fit(documents)
```

After fitting, the vectorizer has learned the vocabulary from the supplied documents.

### 6.2 What Does Transform Mean?

The `transform()` method converts new documents into numerical vectors using the vocabulary already learned during fitting.

Example:

```python
new_documents = [
    "machine science"
]

new_matrix = vectorizer.transform(new_documents)

print(new_matrix.toarray())
```

The vectorizer uses the existing vocabulary instead of learning a new one.

### 6.3 What Does Fit_Transform Mean?

The `fit_transform()` method combines two operations:

1. It learns the vocabulary from the supplied documents.
2. It immediately converts those documents into numerical vectors.

Example:

```python
vectorizer = CountVectorizer()

matrix = vectorizer.fit_transform(documents)
```

It is commonly used when preparing training data.

### 6.4 Difference Between Fit, Transform, and Fit_Transform

| Method | Main Purpose |
|---|---|
| `fit()` | Learns vocabulary and feature information. |
| `transform()` | Converts documents using an already learned vocabulary. |
| `fit_transform()` | Learns the vocabulary and converts documents in one operation. |

### 6.5 What Happens to Out-of-Vocabulary Words?

Out-of-vocabulary (OOV) words are words that were not present in the vocabulary learned during training.

For example:

Training document:

"Machine learning is useful."

New document:

"Machine learning is powerful."

If "powerful" was not in the training vocabulary, a standard `CountVectorizer` will ignore that word during transformation.

It will represent only the words that match the existing vocabulary.

This is one limitation of traditional vocabulary-based vectorizers.

### 6.6 What Is Data Leakage?

Data leakage happens when information from outside the training data is used during model training.

In text vectorization, one example is fitting a vocabulary using both training documents and test documents before evaluating the model.

This allows information from the test set to influence the feature representation.

As a result, the model evaluation may appear better than its true ability to generalize to unseen data.

### 6.7 Why Does Data Leakage Make Evaluation Unreliable?

The purpose of a test dataset is to measure how well a trained model performs on data it has not seen during training.

If information from the test dataset is used to build the vocabulary or select features, the evaluation is no longer completely independent.

This can create an unrealistic estimate of model performance.

### 6.8 Correct Order of Steps

The correct procedure is:

1. Split the original dataset into training and testing sets.
2. Fit the vectorizer using only the training documents.
3. Transform the training documents.
4. Transform the test documents using the same fitted vectorizer.
5. Train the Machine Learning model using the training vectors.
6. Evaluate the model using the test vectors.

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

X_train, X_test = train_test_split(
    documents,
    test_size=0.2,
    random_state=42
)

vectorizer = CountVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)

X_test_vectors = vectorizer.transform(X_test)
```

The important rule is that the vectorizer must be fitted on training data only.

For more complex workflows, a Scikit-learn Pipeline can help ensure that preprocessing and vectorization are performed correctly during model evaluation.

**Conclusion:** `fit()` learns the vocabulary, `transform()` applies it, and `fit_transform()` performs both operations. Correct separation of training and testing data prevents data leakage and produces more trustworthy evaluation results.

---

## Question 7: Case Study – Text Representation for African Language Sentiment Analysis

### 7.1 Research Paper

The selected research paper is:

**"JCT at SemEval-2023 Tasks 12A and 12B: Sentiment Analysis for Tweets Written in Low-resource African Languages using Various Machine Learning and Deep Learning Methods, Resampling, and HyperParameter Tuning."**

Authors:
- Ron Keinan
- Yaakov Hacohen-Kerner

Published in the Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023).

### 7.2 Research Objective

The study investigated how Machine Learning and Deep Learning methods could identify sentiment in social media messages written in low-resource African languages.

The main task was sentiment classification.

The system had to classify each tweet into one of three categories:

- Positive
- Negative
- Neutral

The researchers focused on the challenges of working with African languages that have fewer annotated datasets and fewer NLP resources.

### 7.3 Dataset

The researchers used the AfriSenti-SemEval Twitter datasets.

The study covered 12 African languages, including:

- Hausa
- Yoruba
- Amharic
- Swahili
- Darija
- Algerian Arabic
- Nigerian Pidgin
- Igbo
- Xitsonga
- Twi
- Kinyarwanda
- Mozambique Portuguese

The research included monolingual sentiment classification and multilingual sentiment classification.

The authors also discussed challenges such as limited annotated data, differences in sentiment expressions, cultural context, and class imbalance.

### 7.4 Text Representations Compared

The researchers investigated different ways of representing text.

**A. TF-IDF Word Unigrams**

The system represented individual words using TF-IDF weights.

Word unigrams helped identify important individual terms in tweets.

**B. TF-IDF Character N-Grams**

The researchers also used character sequences ranging from 3 to 10 characters.

Character n-grams can identify useful patterns even when words contain spelling variations or different morphological forms.

**C. BERT Embeddings**

The researchers also experimented with BERT-based representations.

Unlike ordinary TF-IDF features, BERT produces contextualized representations that can capture relationships between words and their surrounding context.

### 7.5 Machine Learning and Deep Learning Methods

The researchers tested several classification algorithms.

| Machine Learning Methods | Deep Learning Methods |
|---|---|
| Random Forest | Multilayer Perceptron |
| Support Vector Classifier | Neural Network with BERT embeddings |
| Logistic Regression | |
| Multinomial Naive Bayes | |

They also investigated preprocessing, feature selection, oversampling, sentiment lexicons, and hyperparameter tuning.

### 7.6 Main Results

The researchers reported several important findings:

1. Character n-grams generally performed better than word n-grams in their classical Machine Learning experiments.
2. The best feature settings differed between languages.
3. Support Vector Machines and Random Forest were among the strongest classical classifiers.
4. Oversampling often improved classification performance.
5. African-language BERT models generally performed better than the general Twitter BERT model in most of their language experiments.
6. Deep Learning models did not outperform classical Machine Learning models in every language.

For example, the Swahili results included:

| Representation | Classifier | Test F1 Score |
|---|---|---:|
| Character 7-grams with TF-IDF | Random Forest | 0.561 |
| Word unigrams with TF-IDF | Support Vector Machine | 0.560 |
| BERT-based representation | AfriBERTa with Neural Network | 0.606 |

These results show that the effectiveness of a representation depends on the language, dataset, and classification method.

The results are specific to the experimental setup of this paper and should not be treated as universal performance guarantees.

### 7.7 What Does This Study Teach Us About the Limitations of Bag of Words and TF-IDF?

Bag of Words and TF-IDF are useful because they convert text into numerical features that traditional Machine Learning algorithms can process.

However, they mainly represent word occurrence, frequency, or local character and word patterns.

They do not independently understand the full meaning of a sentence.

For example:

- "The service is good."
- "The service is not good."

A basic unigram TF-IDF representation may identify the words but does not fully understand how the word "not" changes the sentence's sentiment.

Character n-grams can help recognize spelling and morphological patterns, but they do not automatically provide semantic understanding.

BERT-based representations provide a different approach because they use surrounding context to build representations of words and sentences.

The study therefore demonstrates that traditional text representations can be effective, particularly when data and computing resources are limited, but contextual representations offer additional information about language.

### 7.8 Limitations of the Case Study

The study did not isolate representation type as the only changing variable. It also compared different classifiers, preprocessing methods, feature counts, and training configurations.

Therefore, the reported performance differences cannot be attributed entirely to the text representation.

Nevertheless, the research provides a useful practical comparison of traditional TF-IDF features, character n-grams, and contextual BERT representations for African-language sentiment analysis.

**Conclusion:** The study demonstrates the importance of selecting suitable text representations for low-resource African languages. TF-IDF and n-grams provide useful numerical features, while contextual embeddings can capture additional information about word relationships and meaning.

---

## Overall Conclusion

Text vectorization is an essential part of Natural Language Processing because it converts written language into numerical data that Machine Learning algorithms can process.

One-Hot Encoding represents individual words, Bag of Words counts word occurrences, and TF-IDF assigns weights based on term frequency and document frequency.

Document-Term Matrices organize these features into a consistent structure, while sparse matrix formats help reduce memory requirements.

N-grams preserve some local word order, and vectorizer parameters help control vocabulary size.

The distinction between fitting and transforming is important because the vocabulary should be learned from training data only. This prevents data leakage and supports reliable model evaluation.

Finally, traditional methods such as Bag of Words and TF-IDF are useful but have limitations because they do not fully capture context or semantic meaning. More advanced representations, including contextual embeddings, can address some of these limitations.

---

#

**End of Assignment 3 – Part A**