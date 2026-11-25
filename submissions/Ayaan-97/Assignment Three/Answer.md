## Text Vectorization

### 1. Define Text Vectorization

Text vectorization is the process of converting text documents into numerical values (vectors) that Machine Learning models can understand and process. Most Machine Learning algorithms require every document to have the same number of numerical features because they use mathematical operations on structured data, such as matrices. For example, if a model receives one document with 5 features and another with 8 features, it cannot directly compare them as rows of the same feature matrix. Therefore, a shared vocabulary is created, and every document is represented using the same set of features.

*One-Hot Encoding* represents each word using a vector containing one `1` and zeros in all other positions. For example, if the vocabulary is `[cat, dog, bird]`, the word "cat" becomes `[1, 0, 0]`, while "dog" becomes `[0, 1, 0]`. However, one-hot encoding alone is not enough to represent a whole document because it represents individual words rather than showing how frequently words appear in the document. For example, in a customer review saying "The phone is good, good, and affordable," we need a representation that can account for repeated words and all the words in the review. Techniques such as Bag of Words and TF-IDF can represent the entire document as one fixed-length vector.

### 2. Explain Bag of Words and the Document-Term Matrix

Bag of Words (BoW) is a text representation technique that converts documents into numerical vectors by counting how many times each word appears. The results are organized in a Document-Term Matrix (DTM). In this matrix, each row represents one document, each column represents a unique word in the vocabulary, and each cell contains the frequency of a particular word in a particular document. For example, consider two documents: D1 = "I love Python" and D2 = "I love Java." The vocabulary is `[I, love, Python, Java]`, and the matrix becomes:

| Document | I | love | Python | Java |
| -------- | - | ---- | ------ | ---- |
| D1       | 1 | 1    | 1      | 0    |
| D2       | 1 | 1    | 0      | 1    |

It is called a "bag" because it treats words as separate items and ignores their original order and grammatical structure. Losing word order does little harm in tasks such as spam detection, where the presence of words like "winner," "free," and "prize" can help identify unwanted messages. However, it can cause serious problems in sentiment analysis because "This product is good" and "This product is not good" contain similar words but have opposite meanings. BoW may fail to capture the important effect of the word "not."

### 3. Explain Vocabulary Size and Sparsity

Vocabulary size is the total number of unique words or terms identified from a collection of documents. Sparsity refers to a matrix containing many zero values compared with non-zero values. Real document-term matrices become very large and mostly zero because a collection of documents may contain thousands of unique words, but each individual document uses only a small portion of the entire vocabulary. For example, if a dataset contains 10,000 unique words, a particular document might use only 50 of them. Its vector will therefore contain many zeros. Sparse matrices solve this problem by storing only non-zero values and their positions instead of storing every zero, reducing memory usage and making computation more efficient.

Scikit-learn provides settings such as `min_df`, `max_df`, and `max_features` to control vocabulary size. `min_df=2` removes words that appear in fewer than two documents, while `max_df=0.90` removes words appearing in more than 90% of documents. `max_features=1000` limits the vocabulary to at most 1,000 selected features. Preprocessing also changes vocabulary size: lowercasing combines words such as "Python" and "python"; stopword removal removes common words such as "the" and "is"; and stemming combines related forms such as "playing," "played," and "plays" into a common stem. For example, these techniques can reduce the vocabulary of a collection of customer reviews and make the resulting matrix smaller.

### 4. What Are N-grams?

N-grams are continuous sequences of words or tokens used to represent text while preserving some information about word order. A unigram contains one word, a bigram contains two consecutive words, and a trigram contains three consecutive words. For example, the sentence "I love machine learning" produces these unigrams: `I`, `love`, `machine`, `learning`; these bigrams: `I love`, `love machine`, `machine learning`; and these trigrams: `I love machine`, `love machine learning`. Unlike ordinary Bag of Words, n-grams can capture short phrases and some relationships between neighboring words.

Bigrams can change how a sentence is understood because two words together may express a different meaning from either word separately. For example, the phrase "not happy" has a negative meaning, while "happy" has a positive meaning. A unigram representation may treat "not" and "happy" as separate features, but a bigram can represent "not happy" as one feature. However, larger n-grams increase the number of possible features, require more memory and processing time, and can make the matrix more sparse. For example, using five-word sequences in a small dataset may create many features that appear only once, which can make a Machine Learning model overfit.

### 5. Explain TF-IDF

TF-IDF stands for *Term Frequency–Inverse Document Frequency*. It is a text vectorization technique that measures how important a word is in a particular document compared with all documents in a collection. Term Frequency (TF) measures how often a word appears in a document, while Inverse Document Frequency (IDF) measures how rare the word is across the document collection. The idea of IDF is commonly credited to Karen Spärck Jones, who introduced the statistical interpretation of term specificity in 1972. The textbook formula is:

\\[ IDF(t)=\log\left(\frac{N}{df_t}\right) \\]

Here, \\(N\\) is the total number of documents, and \\(df_t\\) is the number of documents containing the term. A rare word receives a higher IDF score because it can help distinguish one document from another. For example, if the word "computer" appears in 2 out of 100 documents, it receives a higher IDF weight than the word "the," which appears in 95 documents.

Scikit-learn's `TfidfVectorizer` uses a slightly different formula by default because it applies smoothing to avoid zero IDF values and adds one after calculating the logarithm:

\\[ IDF(t)=\log\left(\frac{1+N}{1+df_t}\right)+1 \\]

It also applies L2 normalization by default, meaning that each document vector is scaled to have a length of 1. For example, if a customer review contains the word "excellent" several times and that word is uncommon in the dataset, TF-IDF may assign it a high weight. However, TF-IDF values are weights, not probabilities, because they do not represent the likelihood of an event and are not required to fall between 0 and 1 or add up to 1. Their purpose is to indicate the relative importance of words in documents.

## 6. Explain fit(), transform(), fit_transform()

In Scikit-learn, `fit()` learns information from the training documents, such as the vocabulary and IDF values. `transform()` uses the information already learned by `fit()` to convert new documents into numerical vectors. It does not create a new vocabulary. `fit_transform()` combines both operations: it learns the vocabulary and transforms the documents in one step. For example, if the training documents contain the words "Python," "Java," and "programming," the vectorizer learns these words during `fit_transform()`. When a new document is processed using `transform()`, it uses the same learned vocabulary to create its numerical representation.

Out-of-vocabulary (OOV) words are words that were not present in the vocabulary learned during training. Standard Scikit-learn `CountVectorizer` and `TfidfVectorizer` normally ignore these unknown words during transformation. Data leakage happens when the vectorizer is fitted on the entire dataset, including the testing data, before the training and testing sets are separated. This allows information from the test documents, such as vocabulary and document frequencies, to influence the training process. Consequently, the evaluation may no longer represent performance on completely unseen data and can give an overly optimistic or unreliable result. The correct workflow is to split the dataset first, apply `fit_transform()` only to the training data, apply `transform()` to the testing data, train the model, and finally evaluate its performance.

### 7. Case Study: Comparing Text Representations in an African Language

A research study by Yimam et al. (2021), titled *Introducing Various Semantic Models for Amharic: Experimentation and Evaluation with Multiple Tasks and Datasets*, investigated different text representation methods for Amharic, an African language spoken mainly in Ethiopia. One of the tasks was sentiment classification using approximately 9,400 tweets. The researchers compared traditional machine learning approaches using TF-IDF with other representations, including AmWord2Vec, fastText, FLAIR, and other semantic models. The results showed that TF-IDF combined with Logistic Regression achieved an F1-score of 52.92%, while Role2Vec achieved 58.48%. These results demonstrate that different text representations can produce different performance depending on the language, dataset, and NLP task.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.mdpi.com\&sz=32)

MDPI+1



The study also demonstrates a limitation of Bag of Words and TF-IDF: they mainly represent words according to their frequency and importance, rather than understanding their actual meaning or context. For example, the same word may have different meanings depending on the sentence in which it appears. A frequency-based method may give that word a similar representation in both sentences, even when its meaning changes. Contextual models can use surrounding words to understand these differences more effectively. Therefore, BoW and TF-IDF are useful for simple tasks such as document classification and spam detection, but they are less effective when a task requires understanding context, meaning, or relationships between words.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.mdpi.com\&sz=32)

MDPI+1



### References

1\. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/?utm_source=chatgpt.com)

2\. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. [https://doi.org/10.1108/eb026526](https://doi.org/10.1108/eb026526?utm_source=chatgpt.com)

3\. Robertson, S. (2004). Understanding inverse document frequency: On theoretical arguments for IDF. *Journal of Documentation, 60*(5), 503–520. [https://doi.org/10.1108/00220410410560573](https://doi.org/10.1108/00220410410560573?utm_source=chatgpt.com)

4\. Yimam, S. M., Ayele, A. A., Venkatesh, G., Gashaw, I., & Biemann, C. (2021). Introducing various semantic models for Amharic: Experimentation and evaluation with multiple tasks and datasets. *Future Internet, 13*(11), 275. [https://doi.org/10.3390/fi13110275](https://doi.org/10.3390/fi13110275?utm_source=chatgpt.com)

5\. Scikit-learn. *CountVectorizer Documentation*. [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html?utm_source=chatgpt.com)

6\. Scikit-learn. *TfidfVectorizer Documentation*. [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?utm_source=chatgpt.com)

7\. Scikit-learn. *Working with Text Data*. [https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html?utm_source=chatgpt.com)