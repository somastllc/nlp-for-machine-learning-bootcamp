# **Assignment 3 – Vectorization**

# **Part A – Research**

## **Question 1: Text Vectorization and One-Hot Encoding**

**1. What is text vectorization, and why do machine-learning models need it?**

Text vectorization is the process of converting text into numerical representations so that machine-learning algorithms can process it.

Most traditional machine-learning models, such as Logistic Regression, require numerical input rather than raw text. Therefore, words and documents must be converted into numbers before they can be used for training.

**2. Why must every document have the same number of features?**

Every document must have the same number of features because machine-learning models expect a consistent input structure.

For example, if a dataset has a vocabulary of 100 words, every document must be represented using those same 100 feature positions. Each column must represent the same word for every document.

This allows the model to compare documents and learn patterns consistently.

**3. Explain one-hot encoding with an example.**

One-hot encoding represents each word as a binary vector. Each vector contains one `1` at the position assigned to that word and `0` in all other positions.

For example, consider this vocabulary:

`[bus, late, meeting, ticket]`

| Word | bus | late | meeting | ticket |
|---|---:|---:|---:|---:|
| late | 0 | 1 | 0 | 0 |
| ticket | 0 | 0 | 0 | 1 |

The word `late` is represented as `[0, 1, 0, 0]`.

**4. Give one real-life example of text vectorization.**

A real-life example is a mobile-phone spam detection system.

A message such as:

`WIN a phone prize`

may contain words associated with spam messages. Text vectorization converts these words into numerical features that a machine-learning model can use to identify suspicious messages.

## **Question 2: Bag of Words and the Document-Term Matrix**

**1. What is Bag of Words (BoW)?**

Bag of Words is a text representation technique that converts documents into numerical vectors by counting how many times each vocabulary word appears in a document.

It focuses on word frequency rather than grammar or sentence structure.

**2. What is a document-term matrix?**

A document-term matrix is a numerical table that represents multiple documents using a shared vocabulary.

It has three main components:

- **Rows:** Represent individual documents.
- **Columns:** Represent vocabulary terms.
- **Cells:** Contain the number of times each term appears in a document.

For example:

| Document | free | cash | win |
|---|---:|---:|---:|
| D1 | 1 | 1 | 2 |
| D2 | 1 | 1 | 1 |
| D3 | 0 | 0 | 0 |

Each row represents a document, while each column represents a word.

**3. Why is it called a Bag of Words?**

It is called a Bag of Words because it ignores the original order of words.

For example:

- `free prize`
- `prize free`

Both documents produce the same Bag of Words representation because they contain the same words.

**4. What is one limitation of Bag of Words?**

One major limitation of Bag of Words is that it does not understand word order or context. It treats words as independent features and mainly counts how often each word appears.

For example:

- `The food was good.`
- `The food was not good.`

These two sentences have different meanings. The first sentence expresses a positive opinion, while the second expresses a negative opinion. However, Bag of Words may fail to capture how the word `not` changes the meaning of `good`.

This limitation can negatively affect the accuracy of sentiment analysis.

## **Question 3: Vocabulary Size and Sparsity**

**1. What is a vocabulary in text vectorization?**

A vocabulary is the collection of unique words or terms identified by a text vectorizer from a set of documents.

For example, consider these documents:

- D1: `I love machine learning`
- D2: `I love Python`

The vocabulary is:

`[I, love, machine, learning, Python]`

The vocabulary determines the columns of the document-term matrix.

**2. What is vocabulary size?**

Vocabulary size is the total number of unique terms in the vocabulary.

For example, if a vectorizer identifies 500 unique words, the vocabulary size is 500, and the document-term matrix contains 500 columns.

**3. What is sparsity in a document-term matrix?**

Sparsity occurs when most values in a document-term matrix are zero.

This happens because each document normally contains only a small percentage of the words in the entire vocabulary.

For example, a vocabulary may contain 10,000 words, while an individual document contains only 100 of them. Therefore, most entries in its vector will be zero.

**4. Why are sparse matrices useful?**

Sparse matrices are useful because they reduce memory usage and computational costs.

Instead of storing every zero, sparse matrix formats store non-zero values and their positions.

Scikit-learn uses sparse matrix representations to handle large text datasets efficiently.

**5. Explain the following vocabulary parameters.**

| Parameter | Explanation |
|---|---|
| `min_df` | Removes words that appear in fewer than a specified number of documents. |
| `max_df` | Removes words that appear in more than a specified number or proportion of documents. |
| `max_features` | Limits the vocabulary to a maximum number of selected features. |
| `lowercase` | Converts text to lowercase before tokenization by default. |
| `stop_words` | Removes selected common words from the vocabulary. |

These parameters help control vocabulary size and reduce unnecessary features.

However, preprocessing must be selected carefully. For example, removing the word `not` may change the meaning of a sentence.

## **Question 4: N-grams**

**1. What is an n-gram?**

An n-gram is a continuous sequence of `n` words or tokens extracted from text.

N-grams help preserve some word-order information that basic Bag of Words ignores.

There are three common types:

| Type | Meaning | Example |
|---|---|---|
| Unigram | One word | `good` |
| Bigram | Two consecutive words | `not good` |
| Trigram | Three consecutive words | `the food good` |

**2. How do n-grams improve text representation?**

N-grams capture short word sequences and provide additional context.

For example:

`The service was not helpful.`

A unigram representation contains separate words such as `not` and `helpful`.

A bigram representation can capture the phrase `not helpful`, which communicates a negative meaning.

**3. What are the disadvantages of using larger n-grams?**

Larger n-grams can create several problems:

1. They increase vocabulary size.
2. They produce more sparse matrices.
3. They require additional memory and computational resources.
4. Many longer phrases appear only once, making it difficult for a model to learn general patterns.
5. They may fail to recognize similar phrases with different wording.

Therefore, n-grams should be selected according to the task and dataset.

## **Question 5: TF-IDF, Its Origin, and Scikit-learn Implementation**

**1. What is TF-IDF?**

TF-IDF stands for Term Frequency–Inverse Document Frequency.

It is a numerical weighting technique used to measure how important a word is in a document compared with the entire document collection.

It combines two values:

- **Term Frequency (TF):** Measures how frequently a word appears in a particular document.
- **Inverse Document Frequency (IDF):** Reduces the importance of words that appear in many documents.

A word receives a higher TF-IDF weight when it is frequent in a particular document but relatively rare across the collection.

**2. Who introduced the main ideas behind TF-IDF?**

The development of TF-IDF is connected to important early information-retrieval research.

- **Hans Peter Luhn (1958):** Contributed to the use of term frequency for identifying important words in documents.
- **Karen Spärck Jones (1972):** Developed the statistical interpretation of term specificity, which became an important foundation for inverse document frequency.

Her paper was titled *A Statistical Interpretation of Term Specificity and Its Application in Retrieval*.

**3. Explain the mathematical formula of TF-IDF.**

The common textbook formulas are:

**Term Frequency:**

\[
TF(t,d)=\frac{f_{t,d}}{|d|}
\]

**Inverse Document Frequency:**

\[
IDF(t)=\log\left(\frac{N}{df(t)}\right)
\]

**TF-IDF:**

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

Where:

- \(f_{t,d}\) = Number of times term \(t\) appears in document \(d\).
- \(|d|\) = Total number of tokens in document \(d\).
- \(N\) = Total number of documents.
- \(df(t)\) = Number of documents containing term \(t\).

**4. How does scikit-learn implement TF-IDF differently?**

Scikit-learn's `TfidfVectorizer` uses a slightly different default implementation.

Its smoothed IDF formula is:

\[
IDF(t)=\log\left(\frac{1+N}{1+df(t)}\right)+1
\]

The main differences are:

1. It uses raw term counts by default rather than dividing by document length.
2. It applies IDF smoothing to avoid extreme values.
3. It applies L2 normalization by default to normalize each document vector.

These differences mean that scikit-learn's numerical TF-IDF values may differ from textbook hand calculations.

**5. Are TF-IDF values probabilities?**

No. TF-IDF values are weights, not probabilities.

They measure the relative importance of terms in documents. They do not represent the probability that a document belongs to a particular class.

## **Question 6: Fit, Transform, Out-of-Vocabulary Words, and Data Leakage**

**1. What is the difference between `fit`, `transform`, and `fit_transform`?**

| Method | Explanation |
|---|---|
| `fit()` | Learns the vocabulary and other required information from the input text. |
| `transform()` | Converts text into numerical features using an already learned vocabulary. |
| `fit_transform()` | Learns the vocabulary and transforms the same input in one operation. |

For example, a vectorizer can learn its vocabulary from training documents and then use that vocabulary to convert new documents.

**2. What are out-of-vocabulary (OOV) words?**

Out-of-vocabulary words are words that were not included in the vocabulary learned during the fitting process.

For example, suppose the training vocabulary is:

`[class, free, notes, prize]`

A new document contains:

`free meeting tomorrow`

The words `meeting` and `tomorrow` are unknown to the fitted vectorizer. They are ignored during transformation because the vectorizer has no feature columns for them.

If every word is unknown, the resulting document vector may contain only zeros.

**3. What is data leakage in text vectorization?**

Data leakage occurs when information from the test dataset influences the training process.

For example, if a vectorizer learns its vocabulary from both training and test documents before model evaluation, information from the test data has entered the training preparation process.

This can make the evaluation less reliable because the test data is no longer completely independent.

**4. What is the correct order for fitting and transforming data?**

The correct procedure is:

1. Split the dataset into training and testing sets.
2. Fit the vectorizer using only the training text.
3. Transform the training text.
4. Transform the test text using the same fitted vectorizer.
5. Train and evaluate the machine-learning model.

In cross-validation, the vectorizer should be included inside a Pipeline so that it is fitted separately within each training fold.

## **Question 7: African-Language Case Study – Sentiment Analysis**

**Research Paper:**

Rathi, S., Pande, S., Atkare, H., Tangsali, R., Vyawahare, A., and Kadam, D. (2023).

*Trinity at SemEval-2023 Task 12: Sentiment Analysis for Low-resource African Languages using Twitter Dataset.*

**1. What was the main purpose of the research?**

The research focused on sentiment analysis for three low-resource African languages:

- Hausa
- Yoruba
- Swahili

The main objective was to classify social-media messages into three sentiment categories:

- Positive
- Negative
- Neutral

The study used the AfriSenti-SemEval dataset.

**2. What datasets were used?**

The researchers used Twitter datasets for the three languages.

| Language | Training | Development | Testing |
|---|---:|---:|---:|
| Hausa | 14,172 | 2,677 | 5,303 |
| Yoruba | 8,552 | 2,090 | 4,515 |
| Swahili | 1,810 | 453 | 748 |

These datasets provided labeled examples for training and evaluating sentiment-classification systems.

**3. What methods did the researchers use?**

The study compared traditional machine-learning classifiers with a multilingual deep-learning model.

Traditional machine-learning methods included:

1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Random Forest

The traditional models used TF-IDF with unigrams and bigrams to convert text into numerical features.

The researchers also evaluated multilingual BERT (mBERT), which uses contextual language representations.

They experimented with random oversampling and SMOTE to address class imbalance.

**4. What were the main findings?**

The reported best results differed across the three languages.

| Language | Best reported method | Weighted F1 |
|---|---|---:|
| Hausa | mBERT | 76.53 |
| Yoruba | SVC with SMOTE | 74.83 |
| Swahili | Logistic Regression with SMOTE | 57.79 |

The findings show that no single approach performed best for every language.

mBERT achieved the best reported result for Hausa, while traditional TF-IDF-based machine-learning methods performed best for Yoruba and Swahili in the reported experiments.

**5. What were the limitations of the research?**

The research highlighted limitations related to low-resource African languages.

Some important challenges include:

1. Limited availability of labeled language datasets.
2. Differences in spelling and informal social-media language.
3. Class imbalance in the datasets.
4. Limited availability of language-specific pretrained models and tokenizers.
5. Differences in performance between languages.

These challenges make it difficult to develop equally reliable NLP systems for all African languages.

**6. How does this research relate to text vectorization?**

The research demonstrates how TF-IDF can convert African-language text into numerical features for traditional machine-learning models.

Using unigrams and bigrams allows the system to represent individual words and short word sequences.

However, TF-IDF does not fully understand the meaning or context of words. Contextual models such as mBERT can capture information from surrounding words.

The results also show that traditional machine-learning methods can remain useful for low-resource languages when suitable datasets and preprocessing techniques are available.

**7. What can be learned from this case study?**

This case study demonstrates that text vectorization is an important step in developing NLP applications for African languages.

It also shows that model performance depends on several factors, including:

- Dataset size and quality.
- Language characteristics.
- Text preprocessing.
- Class distribution.
- Choice of machine-learning model.

More research and better language resources are needed to improve NLP systems for low-resource African languages.

## **References**

1. Jones, K. S. (1972). *A Statistical Interpretation of Term Specificity and Its Application in Retrieval*. Journal of Documentation, 28(1), 11–21. https://doi.org/10.1108/eb026526

2. Luhn, H. P. (1958). *The Automatic Creation of Literature Abstracts*. IBM Journal of Research and Development, 2(2), 159–165. https://doi.org/10.1147/rd.22.0159

3. Manning, C. D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

4. Scikit-learn Developers. *Feature Extraction: Text Feature Extraction and TF-IDF Term Weighting*. https://scikit-learn.org/stable/modules/feature_extraction.html

5. Rathi, S., et al. (2023). *Trinity at SemEval-2023 Task 12: Sentiment Analysis for Low-resource African Languages using Twitter Dataset*. Proceedings of SemEval-2023. https://doi.org/10.18653/v1/2023.semeval-1.161

6. Muhammad, S. H., et al. (2023). *AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages*. Proceedings of EMNLP 2023. https://doi.org/10.18653/v1/2023.emnlp-main.862

---

# **Part B – Practical**

### **Task 1: One-Hot Encoding**

One-hot encoding represents each word as a binary vector. A value of `1` means the word is present, while `0` means it is absent. Unlike CountVectorizer, one-hot encoding does not count how many times a word appears. For example, `win` appears twice in D1, but its one-hot value is still `1`.

### **Task 2: Bag of Words and Document-Term Matrix**

The Document-Term Matrix (DTM) represents each document as a row and each vocabulary word as a column. Each cell contains the number of times a word appears in that document.

**Vocabulary (32 words):**

`are, before, but, call, cash, claim, class, entry, for, free, good, great, lecture, lunch, me, meeting, not, notes, now, phone, please, prizes, reply, send, still, the, to, today, was, we, were, win`

**Document-Term Matrix:**

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

The matrix has 5 rows and 32 columns, giving a shape of `(5, 32)`. The word `win` appears twice in D1, so its count is `2`. The DTM shows word frequency, but it does not understand the meaning or context of words.

### **Task 3: Tokens vs. Vocabulary Size**

The five documents contain 41 total tokens and 32 unique vocabulary words. The number of tokens in each document is D1 = 8, D2 = 8, D3 = 7, D4 = 8, and D5 = 10. The words appearing in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win`.

### **Task 4: Preprocessing Inside the Vectorizer**

CountVectorizer provides options for controlling text preprocessing. By default, it converts text to lowercase and ignores punctuation. It also ignores single-character tokens because its default token pattern requires at least two characters. When `lowercase=False` is used, uppercase and lowercase words are treated as different tokens. The English stop-word list removes common words such as `now`, `to`, and `not`, which can sometimes remove important meaning from a sentence.

### **Task 5: Limiting the Vocabulary**

The `min_df=2` parameter keeps only words that appear in at least two documents, reducing the vocabulary from 32 to 7 words: `free`, `lecture`, `notes`, `now`, `the`, `today`, and `win`. The `max_features=5` parameter limits the vocabulary to five selected features: `free`, `lecture`, `now`, `the`, and `win`. These parameters can reduce the size of the feature matrix, but they may also remove useful words.

### **Task 6: N-grams**

Unigrams represent individual words, while bigrams represent pairs of consecutive words. For D5, the unigram features include `but`, `good`, `great`, `lecture`, `not`, `notes`, `the`, `was`, and `were`. Using both unigrams and bigrams allows the model to capture short phrases such as `not good` and `were great`. In this dataset, the vocabulary sizes are 32 for unigrams, 67 for unigrams and bigrams, and 98 for unigrams, bigrams, and trigrams.

### **Task 7: TF-IDF**

The following table compares raw word counts with TF-IDF values for every word in D2.

| Word | Count | TF-IDF |
|---|---:|---:|
| call | 1 | 0.389141 |
| cash | 1 | 0.389141 |
| entry | 1 | 0.389141 |
| free | 1 | 0.313957 |
| now | 1 | 0.313957 |
| prizes | 1 | 0.389141 |
| today | 1 | 0.313957 |
| win | 1 | 0.313957 |

CountVectorizer records how many times a word appears, while TfidfVectorizer considers both word frequency and how common the word is across all documents. Words that are less common across the document collection can receive higher TF-IDF scores. Therefore, TF-IDF can give more importance to words that help distinguish one document from another.

### **Task 8: TF-IDF by Hand**

For D2: `Free entry: win cash prizes today. Call now.`

The document contains 8 tokens.

**Formulas:**

- TF = Word count / Total words in document
- IDF = ln(Total number of documents / Number of documents containing the word)
- TF-IDF = TF × IDF

**Calculation for the word `cash`:**

- Count = 1
- TF = 1 / 8 = 0.125
- DF = 1
- IDF = ln(5 / 1) = 1.609438
- TF-IDF = 0.125 × 1.609438 = **0.201180**

**Calculation for the word `today`:**

- Count = 1
- TF = 1 / 8 = 0.125
- DF = 2
- IDF = ln(5 / 2) = 0.916291
- TF-IDF = 0.125 × 0.916291 = **0.114536**

The word `cash` receives a higher manual TF-IDF score than `