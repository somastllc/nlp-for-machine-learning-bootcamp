# Assignment 3: Vectorization

This assignment focuses on how text is converted into numerical representations that Machine Learning models can process.

## Topics Covered

- One-hot encoding
- Bag of Words (BoW)
- Document-Term Matrix (DTM)
- Vocabulary and vocabulary size
- Sparse matrices
- N-grams, including unigrams, bigrams, and trigrams
- TF-IDF and inverse document frequency (IDF)
- `CountVectorizer`
- `TfidfVectorizer`
- `fit()`, `transform()`, and `fit_transform()`
- Out-of-vocabulary (OOV) words
- Data leakage during vectorization
- Limitations of Bag of Words and TF-IDF
- Cosine similarity

## Practical Work

The practical tasks applied different vectorization methods to sample messages and reviews. They included building a custom one-hot encoder, creating a Document-Term Matrix, comparing n-gram vocabularies, calculating TF-IDF manually and with scikit-learn, analyzing sparse matrices, handling unknown words, demonstrating data leakage, and testing the limitation of count-based representations using cosine similarity.

## Main Learning

The assignment showed how text can be represented as numerical features for Machine Learning, while also demonstrating that methods such as Bag of Words and TF-IDF mainly capture word occurrence and frequency rather than the actual meaning or semantic relationships between words.