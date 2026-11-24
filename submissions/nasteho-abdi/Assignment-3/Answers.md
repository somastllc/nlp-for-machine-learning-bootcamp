Assignment 3 – Vectorization

Course: NLP for Machine Learning Bootcamp
Assignment: Assignment 3 - vectorization
Prepared by: Nasteho Abdi Aden

Part A – Research

1. Definition of Text Vectorization

Text vectorization is the process of converting text documents into numerical representations that computers and Machine Learning algorithms can understand and process.

Most Machine Learning models work with numerical data rather than raw text. Therefore, words, sentences, and documents must be transformed into numerical features before they can be used for training and prediction.

Most Machine Learning models require every document to have the same number of numerical features because mathematical operations such as matrix multiplication and distance calculations require consistent dimensions. For example, if one document is represented by 100 features and another by 80 features, they cannot be directly compared using the same feature space.

One-Hot Encoding of Words

One-hot encoding is a technique that represents each word using a binary vector. Each word receives a unique position in a vocabulary, where its position contains 1 and all other positions contain 0.

For example, consider the vocabulary:

"[cat, dog, bird]"

The representations are:

- cat = [1, 0, 0]
- dog = [0, 1, 0]
- bird = [0, 0, 1]

However, one-hot encoding is not sufficient to represent an entire document because it represents individual words separately. A document containing several words requires multiple vectors or an additional method to combine them.

Real-Life Example

In an email spam detection system, the words "free", "winner", and "prize" may be useful for identifying suspicious messages. One-hot encoding can identify these words, but it does not explain how frequently they appear or how they relate to other words in the email.

Therefore, techniques such as Bag of Words and TF-IDF are used to represent complete documents.

2. Bag of Words and Document-Term Matrix

Bag of Words (BoW) is a text representation technique that converts documents into numerical vectors based on the frequency of words appearing in each document.

It creates a vocabulary containing the unique words found in the entire collection of documents. Each document is then represented by counting how many times each vocabulary word appears.

It is called a "bag" because it treats words as if they were placed inside a bag without preserving their original order or grammatical structure.

Document-Term Matrix

A document-term matrix is a table that represents the frequency of vocabulary words across multiple documents.

It contains three main components:

- Rows: Represent individual documents.
- Columns: Represent unique words in the vocabulary.
- Cells: Represent the number of times each word appears in a document.

Example

Consider these documents:

- D1: I love machine learning.
- D2: I love Python.

The document-term matrix would look like this:

Document| I| love| machine| learning| Python
D1| 1| 1| 1| 1| 0
D2| 1| 1| 0| 0| 1

Advantages and Limitations

Bag of Words is simple, efficient, and useful for many text classification tasks. For example, it can help classify emails into spam and non-spam categories based on the presence of certain words.

However, it loses word order, which can cause problems when sentence meaning depends on the arrangement of words.

For example:

- "The dog bites the man."
- "The man bites the dog."

Both sentences contain the same words, but their meanings are different.

Losing word order causes little harm in some tasks, such as basic document topic classification, but it can cause serious problems in sentiment analysis and language understanding.

3. Vocabulary Size and Sparsity

Vocabulary refers to the collection of all unique words identified in a text corpus.

Vocabulary size is the total number of unique words in that collection.

For example, consider:

- Document 1: I love Python.
- Document 2: I love Machine Learning.

The vocabulary is:

"[I, love, Python, Machine, Learning]"

The vocabulary size is 5.

Sparsity

Sparsity refers to the large number of zero values in a document-term matrix.

Real-world text datasets often contain thousands or millions of unique words. However, an individual document usually contains only a small fraction of the complete vocabulary.

For example, a document may contain 20 words while the entire dataset has 50,000 unique words. Therefore, most cells in its document-term vector will contain zero.

Why Sparse Matrices Are Important

Storing every zero value wastes memory and computational resources.

Sparse matrix formats such as Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) store non-zero values together with their positions instead of storing every cell.

This makes large text datasets more memory-efficient and faster to process.

Settings That Change Vocabulary Size

Several CountVectorizer settings influence the vocabulary:

- "min_df": Removes words that appear in fewer than a specified number of documents.
- "max_df": Removes words that appear in more than a specified proportion or number of documents.
- "max_features": Limits the vocabulary to a maximum number of features, usually selected according to term frequency.

Preprocessing also changes vocabulary size:

- Lowercasing: Converts uppercase and lowercase forms into the same word. For example, "Python" and "python" become "python".
- Stopword removal: Removes common words such as "the", "is", and "and".
- Stemming: Reduces related word forms to a common stem. For example, "playing", "played", and "plays" may be reduced to "play".

These techniques reduce unnecessary vocabulary growth and can improve model efficiency.

4. N-grams

N-grams are continuous sequences of words used to represent text while preserving some of its original word order.

There are three common types:

Unigrams

Unigrams contain individual words.

Example:

Sentence: "Machine learning is useful."

Unigrams:

"Machine, learning, is, useful"

Bigrams

Bigrams contain sequences of two consecutive words.

Example:

"Machine learning, learning is, is useful"

Trigrams

Trigrams contain sequences of three consecutive words.

Example:

"Machine learning is, learning is useful"

How N-grams Preserve Meaning

Consider the sentence:

"I do not like this product."

A unigram representation contains individual words such as "not" and "like", but does not preserve their relationship.

A bigram representation contains "not like", which helps identify the negative expression.

Therefore, bigrams can provide useful information that ordinary Bag of Words may lose.

Cost of Larger N-grams

Larger n-gram ranges generate more features because they create additional combinations of consecutive words.

This can increase:

- Vocabulary size.
- Memory consumption.
- Computational cost.
- Risk of overfitting, especially with small datasets.

Therefore, the n-gram range should be selected according to the task and available training data.

5. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF is a numerical weighting technique used in Natural Language Processing to estimate how important a word is to a particular document compared with an entire collection of documents.

It combines two measurements:

Term Frequency (TF)

TF measures how frequently a word appears in a document.

A common textbook formula is:

TF(t,d) = Count of term t in document d / Total number of terms in document d

Inverse Document Frequency (IDF)

IDF measures how rare or informative a word is across the document collection.

The textbook formula is:

IDF(t) = log(N / df(t))

Where:

- N = Total number of documents.
- df(t) = Number of documents containing term t.

TF-IDF Formula

TF-IDF(t,d) = TF(t,d) × IDF(t)

The idea of inverse document frequency is associated with Karen Spärck Jones, who published her influential research paper, "A Statistical Interpretation of Term Specificity and Its Application in Retrieval", in 1972.

Her research explained that words appearing in fewer documents can provide more useful information for distinguishing documents than words appearing everywhere.

Why Rare Words Receive Higher Weights

Words that appear in many documents usually provide less information about the specific content of an individual document.

For example, words such as "the", "is", and "and" occur frequently in English documents.

However, a word such as "cybersecurity" may be more useful for identifying documents about cybersecurity.

Therefore, rare words generally receive higher IDF values.

How Scikit-learn's TfidfVectorizer Differs

Scikit-learn's "TfidfVectorizer" uses a smoothed IDF formula by default:

IDF(t) = log((1 + N) / (1 + df(t))) + 1

It also applies L2 normalization by default.

This means that the resulting document vector is divided by its Euclidean length so that its overall L2 norm becomes 1.

Other default settings include:

- "smooth_idf=True"
- "norm='l2'"
- "sublinear_tf=False"

These settings make the implementation slightly different from the basic textbook formula.

TF-IDF Values Are Not Probabilities

TF-IDF values are numerical weights that indicate the relative importance of words in documents.

They are not probabilities because they do not represent the likelihood of an event or sum to one as a probability distribution.

Their purpose is to measure the importance of terms for text representation and classification.

6. Fit, Transform, and Data Leakage

In Machine Learning, vectorization involves three important operations: fit, transform, and fit_transform.

Fit

The fit operation learns the vocabulary and other necessary statistics from the training documents.

For example, a CountVectorizer learns which unique words appear in the training corpus.

Transform

The transform operation converts new documents into numerical vectors using the vocabulary learned during fitting.

It does not create a new vocabulary.

Fit_transform

The fit_transform operation combines both operations. It learns the vocabulary and immediately converts the training documents into numerical vectors.

Out-of-Vocabulary Words

Out-of-vocabulary (OOV) words are words that were not included in the vocabulary learned during training.

When a new document contains an unknown word, CountVectorizer ignores that word during transformation because no feature position exists for it.

For example, if the training vocabulary contains "prizes" but a new document contains "prize", the word "prize" will be treated as unknown.

Stemming or lemmatization can help reduce this problem by converting related word forms into a common representation.

Data Leakage in Vectorization

Data leakage occurs when information from the test dataset is used during model training or preprocessing.

For example, fitting a vectorizer on both training and test documents allows the vocabulary to include words from the test set.

This creates an unfair evaluation because information from the test data has influenced the training process.

As a result, evaluation performance may appear better than it would be on genuinely unseen data.

Correct Order of Steps

The correct procedure is:

1. Split the dataset into training and testing sets.
2. Fit the vectorizer using only the training data.
3. Transform the training data.
4. Transform the test data using the same fitted vectorizer.
5. Train the Machine Learning model using the training vectors.
6. Evaluate the model using the test vectors.

This procedure prevents vocabulary leakage and provides a more reliable evaluation.

7. Case Study: Swahili News Classification Using Different Text Representations

A useful research study is "Graph Convolutional Network for Swahili News Classification" by Alexandros Kastanos and Tyler Martin (2021).

The research investigates how different text representation methods can be used to classify Swahili news articles, particularly when only a limited number of labelled documents are available.

Task

The main task was to automatically classify Swahili news articles into six categories:

- National news.
- Sports.
- Entertainment.
- Economy.
- International news.
- Health.

Dataset

The researchers used a Swahili News Classification dataset containing 23,266 labelled samples.

The dataset was divided into three subsets:

- 80% training data.
- 10% validation data.
- 10% testing data.

The researchers also applied text preprocessing, including stemming.

Representations Compared

The study compared several text representation methods:

1. Count-based representation (Bag of Words): Represents documents using word frequencies.
2. TF-IDF: Assigns weights to words according to their importance across documents.
3. fastText: Uses pre-trained word embeddings to represent words as dense numerical vectors.
4. PV-DBOW and PV-DM: Document embedding methods based on Doc2Vec.
5. Text Graph Convolutional Network (Text GCN): Uses relationships between documents and words to learn representations.

Results

The reported test results included:

Representation/Model| Accuracy| Macro F1
TF-IDF| 83.07%| 68.72%
Counts| 83.32%| 73.60%
fastText| 67.47%| 32.41%
PV-DBOW| 81.64%| 72.93%
PV-DM| 77.01%| 67.50%
Text GCN| 84.62%| 75.29%
Text GCN-t2v| 85.40%| 75.67%

The results show that the Text GCN-t2v model achieved the highest reported performance among the compared methods.

Discussion and Limitations

The study demonstrates that traditional text representations such as Bag of Words and TF-IDF can provide useful results for text classification.

However, these methods mainly represent word frequencies and statistical importance rather than the actual meanings and relationships between words.

For example, words with similar meanings may receive completely different representations if they have different spellings. Similarly, words with different meanings may be treated as separate features without understanding their context.

Word embeddings and graph-based approaches provide additional information about relationships between words and documents.

The study also demonstrates that more advanced representations can be useful when working with low-resource African languages such as Swahili.

Conclusion

The research shows that text representation plays an important role in NLP classification tasks. Bag of Words and TF-IDF are useful baseline methods, but they have limitations in representing semantic relationships. More advanced approaches can provide additional contextual and relational information.

References

1. Sparck Jones, K. (1972). A Statistical Interpretation of Term Specificity and Its Application in Retrieval. Journal of Documentation, 28(1), 11–21. https://doi.org/10.1108/eb026526

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. https://nlp.stanford.edu/IR-book/

3. Scikit-learn Developers. (2026). Feature Extraction: Text Feature Extraction. https://scikit-learn.org/stable/modules/feature_extraction.html

4. Scikit-learn Developers. (2026). TfidfVectorizer Documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

5. Kastanos, A., & Martin, T. (2021). Graph Convolutional Network for Swahili News Classification. AfricaNLP Workshop. https://arxiv.org/abs/2103.09325

6. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

Part B – Practical Report :

Task 1 – One-Hot Encoding

The vocabulary contains six words: cash, class, free, lunch, notes, and win. Each word is represented by a binary vector containing one value of 1 and the remaining values as 0.

The word "free" is represented as "[0, 0, 1, 0, 0, 0]", while "cash" is represented as "[1, 0, 0, 0, 0, 0]".

For the sentence "win free cash", three vectors are generated because the sentence contains three words. Each vector has six numerical features because the vocabulary contains six words.

Task 2 – Bag of Words

Document-Term Matrix

The following table shows the word frequencies for the five training documents. Each row represents a document, each column represents a vocabulary word, and each cell represents the number of occurrences of that word.

Document| are| before| but| call| cash| claim| class| entry
D1| 0| 0| 0| 0| 0| 1| 0| 0
D2| 0| 0| 0| 1| 1| 0| 0| 1
D3| 1| 0| 0| 0| 0| 0| 0| 0
D4| 0| 1| 0| 0| 0| 0| 1| 0
D5| 0| 0| 1| 0| 0| 0| 0| 0

Document| for| free| good| great| lecture| lunch| me| meeting
D1| 0| 1| 0| 0| 0| 0| 0| 0
D2| 0| 1| 0| 0| 0| 0| 0| 0
D3| 1| 0| 0| 0| 0| 1| 0| 1
D4| 0| 0| 0| 0| 1| 0| 1| 0
D5| 0| 0| 1| 1| 1| 0| 0| 0

Document| not| notes| now| phone| please| prizes| reply| send
D1| 0| 0| 1| 1| 0| 0| 1| 0
D2| 0| 0| 1| 0| 0| 1| 0| 0
D3| 0| 0| 0| 0| 0| 0| 0| 0
D4| 0| 1| 0| 0| 1| 0| 0| 1
D5| 1| 1| 0| 0| 0| 0| 0| 0

Document| still| the| to| today| was| we| were| win
D1| 0| 0| 1| 0| 0| 0| 0| 2
D2| 0| 0| 0| 1| 0| 0| 0| 1
D3| 1| 0| 0| 1| 0| 1| 0| 0
D4| 0| 1| 0| 0| 0| 0| 0| 0
D5| 0| 2| 0| 0| 1| 0| 1| 0

Questions and Answers

1. The document-term matrix has 5 rows and 32 columns, representing five documents and 32 unique vocabulary words. The value of "win" in D1 is 2 because the word appears twice in that document.

2. The vectors for "win free phone" and "phone free win" are identical because Bag of Words counts word frequencies without considering word order.

Task 3 – Tokens vs Vocabulary Size

The total number of token occurrences in D1–D5 is 41, while the vocabulary size is 32 unique words.

The words appearing in more than one document are:

Word| Number of Documents
free| 2
lecture| 2
notes| 2
now| 2
the| 2
today| 2
win| 2

This shows that the total number of tokens is different from vocabulary size because some words appear multiple times across the documents.

Task 4 – Preprocessing Inside the Vectorizer

The default CountVectorizer converts all text into lowercase, so uppercase and lowercase versions of the same word are treated as one vocabulary entry.

When "lowercase=False" is used, uppercase and lowercase forms remain separate. For example, "WIN" and "Win" become different vocabulary entries.

The single-letter word "a" and punctuation marks such as "!" are not included because the default token pattern ignores single-character tokens and punctuation.

Using "stop_words="english"" removes common English words such as "now" and "to" from D1.

The word "not" is also removed from the vocabulary when English stopwords are enabled. This is important because D5 contains the expression "not good". Removing "not" can change the sentiment information and make negative sentences more difficult to identify correctly.

Task 5 – Limiting the Vocabulary

Using "min_df=2" produces a vocabulary of 7 words:

"['free', 'lecture', 'notes', 'now', 'the', 'today', 'win']"

These words appear in at least two different documents.

Using "max_features=5" produces a vocabulary of 5 words:

"['free', 'lecture', 'now', 'the', 'win']"

This setting limits the vocabulary to five selected features according to their overall frequency. Vocabulary limitation reduces the number of features and can make text processing more efficient.

Task 6 – N-grams

Unigrams represent individual words, while bigrams represent sequences of two consecutive words. Trigrams represent sequences of three consecutive words.

In D5, the important bigram is "not good". It preserves the relationship between the negative word "not" and the word "good", which helps identify the negative opinion.

The vocabulary sizes on D1–D5 are:

N-gram range| Vocabulary size
(1, 1)| 32
(1, 2)| 67
(1, 3)| 98

The results show that vocabulary size increases as larger n-grams are included. Larger n-grams preserve more word-order information but require additional memory and computational resources.

Task 7 – TF-IDF

TF-IDF measures the importance of a word in a document compared with the entire document collection.

The following table compares word counts and TF-IDF weights for D2.

Word| Count| TF-IDF Weight
call| 1| 0.3891
cash| 1| 0.3891
entry| 1| 0.3891
free| 1| 0.3140
now| 1| 0.3140
prizes| 1| 0.3891
today| 1| 0.3140
win| 1| 0.3140

Although every word in D2 appears once, their TF-IDF weights are different because their document frequencies are different.

Words such as "cash", "call", "entry", and "prizes" appear in only one document, so they receive higher inverse document frequency values.

In D1, the word "win" receives the highest weight of 0.5739 because it appears twice in the document and therefore has a higher term frequency than the other words.

Task 8 – TF-IDF by Hand

The textbook formula is:

TF = Term Count / Total Number of Terms

IDF = log(N / df)

TF-IDF = TF × IDF

For this calculation:

- Total number of documents (N) = 5.
- Total number of words in D2 = 8.

Calculation for the word "cash"

Count = 1

Document frequency = 1

TF = 1 / 8 = 0.1250

IDF = log(5 / 1) = 1.6094

TF-IDF = 0.1250 × 1.6094

TF-IDF = 0.2012

Scikit-learn TF-IDF = 0.3891

Calculation for the word "today"

Count = 1

Document frequency = 2

TF = 1 / 8 = 0.1250

IDF = log(5 / 2) = 0.9163

TF-IDF = 0.1250 × 0.9163

TF-IDF = 0.1145

Scikit-learn TF-IDF = 0.3140

Comparison

Both methods rank "cash" higher than "today". However, their numerical values are different because scikit-learn applies IDF smoothing and L2 normalization by default, while the textbook calculation above uses the basic formula without normalization.

Task 9 – Sparse Matrix

The CountVectorizer matrix contains 160 total cells because it has 5 documents and 32 vocabulary features.

Measurement| Result
Total cells| 160
Non-zero cells| 39
Zero cells| 121
Percentage of zeros| 75.62%
Storage format| CSR

The matrix contains 121 zero values, representing 75.62% of all cells. This demonstrates that document-term matrices are sparse because most documents contain only a small portion of the complete vocabulary. CSR (Compressed Sparse Row) stores non-zero values efficiently and reduces memory usage.

Task 10 – Fit and Transform

The CountVectorizer was fitted using only D1–D4. It learned a vocabulary of 26 unique words.

The new documents were transformed using this existing vocabulary.

The out-of-vocabulary words are:

- N1: "your", "prize".
- N2: "is", "moved", "tomorrow".

These words are ignored because they were not included in the training vocabulary.

The word "prize" is unknown because the training document contains "prizes", which is a different word form.

Stemming or lemmatization from preprocessing can help solve this problem by reducing related word forms to a common representation.

Task 11 – Data Leakage

Data leakage occurs when information from the test dataset is used during training or preprocessing.

When the vectorizer is fitted only on D1–D4, the vocabulary contains 26 words. When it is fitted on D1–D4 together with N1 and N2, the vocabulary increases to 31 words.

The words leaked from the test set are:

"['is', 'moved', 'prize', 'tomorrow', 'your']"

This demonstrates why a vectorizer must be fitted only on training data. Using test data during fitting allows information from unseen documents to influence the training vocabulary and can produce an unfair evaluation.

Task 12 – Cosine Similarity and Meaning

Cosine similarity measures the similarity between two numerical vectors based on the angle between them.

The reviews are:

- R1: the food was tasty.
- R2: the food was delicious.
- R3: the food was terrible.

The cosine similarity results are:

Reviews| Similarity
R1 vs R2| 0.75
R1 vs R3| 0.75
R2 vs R3| 0.75

R1 is equally similar to R2 and R3 according to the CountVectorizer representation.

This happens because Bag of Words represents word occurrences rather than actual meanings. Although "tasty" and "delicious" have similar positive meanings, the vectorizer treats them as separate words. Similarly, it does not understand that "terrible" expresses a negative opinion.

Therefore, Bag of Words and TF-IDF cannot fully capture semantic meaning.

Practical Conclusion – Choosing a Vectorizer for Spam Detection

For a spam detection system, I would use TF-IDF together with word counts and a suitable n-gram range such as (1, 2). TF-IDF can give higher weights to informative words such as "cash", "prizes", and "claim", while bigrams can identify useful expressions such as "free phone" and "claim win". I would also use lowercasing to combine uppercase and lowercase word forms. However, I would avoid removing all English stopwords without checking their importance because words such as "not" can change the meaning of a message. I would use vocabulary limits carefully to remove extremely rare or uninformative terms without losing important spam indicators. Finally, I would fit the vectorizer only on training data to prevent data leakage.

Final Practical Summary

The practical exercises demonstrated how text can be converted into numerical representations using One-Hot Encoding, Bag of Words, N-grams, and TF-IDF.

The experiments also showed that document-term matrices are often sparse, that vocabulary selection affects the resulting features, and that fitting a vectorizer on training data only is necessary to prevent data leakage.

Finally, the cosine similarity experiment demonstrated an important limitation of traditional text vectorization techniques: they represent word frequencies but do not fully understand the meaning or sentiment of words.