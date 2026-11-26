# Assignment 3 – Vectorization

## Part A – Research

### 1. Text vectorization and one-hot encoding

Text vectorization is the process of representing written language as numerical features. Most traditional machine learning models, such as logistic regression and support vector machines, expect each sample to have the same feature dimensions. This allows a model to associate each column with a consistent feature, regardless of how long the original document is. A fixed-size representation is not a universal requirement for all neural architectures, but it is central to the traditional document-feature approach [1, 2].

Imagine a small grocery shop that records customer requests using the vocabulary `[bread, milk, rice]`. The word `milk` is represented by `[0, 1, 0]`, while `rice` becomes `[0, 0, 1]`. This is one-hot encoding: a word has a single active position corresponding to its identity. A request for "milk rice" produces two vectors, while "bread milk rice" produces three. Although each word vector has the same width, the sequence has a variable number of rows. To represent an entire request with fixed dimensions, we could sum the word vectors into a count vector. One-hot encoding alone also gives unrelated words and synonyms equally distinct identities.

### 2. Bag of Words and the document-term matrix

Bag of Words (BoW) represents a document by the frequency of each vocabulary term. The name comes from treating words as items in a bag: their quantities matter, but their original arrangement does not [1].

For example, consider two short library requests:

| Document | history | math | textbook |
|---|---:|---:|---:|
| math math textbook | 0 | 2 | 1 |
| history textbook | 1 | 0 | 1 |

Each row represents a document, each column represents a vocabulary term, and each cell contains the frequency of that term in that document. A zero means the term is absent. This structure is called a document-term matrix.

For broad topic classification, such as separating sports news from financial news, losing word order may not cause much harm because topic-related terms provide strong clues. However, in a medical statement, "patient has no fever" and "patient has fever" differ in a critical way. An even clearer order-based example is "dog bites man" versus "man bites dog": both have the same unigram counts but describe different events.

### 3. Vocabulary size and sparsity

Vocabulary size is the number of distinct features retained after processing a corpus. A collection of thousands of articles may contain tens of thousands of distinct terms, although any individual article uses only a small fraction of them. Consequently, most entries in its document-term matrix are zero. This property is called sparsity.

Compressed sparse row (CSR) storage avoids keeping every zero. It maintains an array of stored values, their column indices, and row pointers that mark the beginning and end of each row's entries. This is more memory-efficient than a dense matrix when relatively few cells are occupied [3].

| Setting or preprocessing | Effect on vocabulary |
|---|---|
| `min_df` | Removes terms appearing in fewer documents than the specified threshold. |
| `max_df` | Removes terms appearing in more documents than the specified threshold. |
| `max_features` | Keeps at most the specified number of features, ordered by corpus term frequency. |
| Lowercasing | Merges capitalization variants such as Book and BOOK. |
| Stopword removal | Excludes selected common words, which may also remove useful negation. |
| Stemming | Reduces related word forms to a common stem, potentially merging distinct meanings. |

Document-frequency thresholds can be integer document counts or fractions of the corpus. These choices should be tested rather than assumed to improve performance [4].

### 4. N-grams

An n-gram is a contiguous sequence of n tokens. For the sentence "delivery never arrived", the unigrams are `delivery`, `never`, and `arrived`; the bigrams are `delivery never` and `never arrived`; the trigram is `delivery never arrived`. The bigram `never arrived` provides a useful negative signal that the separate word `arrived` cannot provide on its own.

N-grams preserve local order, not the complete grammatical structure of a document. Longer phrases create more distinct features, increase memory and computational requirements, and often occur too rarely to generalize well. Word and character n-grams are both possible, and the appropriate range depends on the task [2].

### 5. TF-IDF

Term Frequency–Inverse Document Frequency balances how frequently a term occurs in a document against how widely it occurs across the collection. Karen Spärck Jones introduced the influential term-specificity idea behind IDF in her 1972 information retrieval paper [5]. A term that occurs in relatively few documents can be more useful for distinguishing documents than one that occurs almost everywhere.

A standard textbook formulation is:

- TF(t,d) = count(t,d)
- IDF(t) = ln(N / df(t))
- TF-IDF(t,d) = TF(t,d) × IDF(t)

Here N is the number of documents and df(t) counts documents containing t. Another common TF convention divides the count by the number of tokens in the document. I use that length-normalized convention in the practical hand calculation and also show the raw-count alternative. The log base can vary; natural logarithms are used here [1].

By default, scikit-learn uses raw term counts (`sublinear_tf=False`) and the smoothed formula `idf(t) = ln((1 + N)/(1 + df(t))) + 1`. It multiplies the counts by IDF and then applies L2 normalization, dividing each document vector by its Euclidean length. Smoothing and the added constant keep the weighting well behaved and give even universally occurring terms a positive IDF. L2 normalization controls the effect of document length [2, 6]. TF-IDF values are weights, not probabilities: their sum is not required to equal one, and they do not estimate the likelihood of a particular word or class.

### 6. Fitting, transforming and data leakage

`fit` learns the vocabulary and, for TF-IDF, document-frequency-based weights. `transform` uses these learned features to encode documents without learning a new vocabulary. `fit_transform` combines both operations and is convenient for training data.

An out-of-vocabulary word is ignored by a fitted CountVectorizer or TfidfVectorizer. A document containing only unknown terms therefore becomes a zero vector. Consistent stemming, lemmatization or character n-grams can reduce some forms of vocabulary mismatch.

Data leakage happens when information that should remain unavailable during training influences the model or preprocessing. Fitting on test documents exposes their vocabulary and can also change IDF weights and feature selection. It can make evaluation optimistically biased, though a score increase is not guaranteed. The correct sequence is to split the raw documents, fit the vectorizer on training data, transform training and test data using that same vectorizer, train the classifier, and evaluate on the held-out test set. During cross-validation, fitting must happen separately inside each training fold, preferably using a pipeline [7].

### 7. Case study: sentiment analysis in African languages

The 2023 SemEval JCT study [8] investigated sentiment classification of Twitter messages in 12 African languages, including Swahili, Hausa, Igbo and Amharic, as well as a combined multilingual dataset. The task was to distinguish positive, negative and neutral sentiment using labeled training, development and test sets. The researchers tested TF-IDF-based word unigrams and character n-grams, sometimes supplemented by sentiment lexicons, with classical classifiers. They also experimented with neural classifiers using BERT representations.

Selected test F1 scores from Tables 4 and 5 of the paper illustrate the comparison:

| Language | TF-IDF-based SVM | BERT-based AfriBERTa | 
|---|---:|---:|
| Hausa | 0.757 | 0.791 |
| Igbo | 0.791 | 0.785 |
| Amharic | 0.360 | 0.796 |

The results are mixed: contextual representations performed much better in Amharic, while a classical system was competitive in Igbo. The experiments differ in classifiers, settings and additional features, so these numbers do not isolate the effect of vectorization alone. The authors also explain that sentiment requires context and cultural knowledge. In my interpretation, this is where count-based representations are limited: related words occupy different columns and frequency alone cannot resolve sarcasm or contextual meaning. Nevertheless, the study shows that simple features can remain useful for low-resource languages. I selected this African-language paper rather than a Somali-only study because it directly compares feature-based and contextual approaches.

## Part B – Practical

The following results were produced by running `vectorization_practice.py`. The complete console output, including library versions, is available in `output.txt`.

### Task 1 – One-hot encoding

Each word has a vector of six positions. The sentence `free cash` produces two vectors, while `win free cash` produces three, because the encoder creates one vector per token. The individual vector width stays fixed, but the number of vectors changes with sentence length.

### Task 2 – Bag of Words

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

The D1 cell for `win` contains 2 because `Win` and `WIN` become the same lowercase token. The two reordered phrases produce identical vectors since both contain the same three terms once. This demonstrates that unigram BoW ignores order.

### Task 3 – Tokens and vocabulary

There are 41 analyzed tokens and 32 distinct vocabulary terms. The words appearing in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today` and `win`. Token count measures occurrences, whereas vocabulary size measures distinct features.

### Task 4 – Preprocessing

The default token pattern excludes the single-letter word `a` and punctuation such as `!`, while lowercasing combines `Win` and `WIN`; disabling lowercasing keeps both forms separate. The English stopword list removes `now` and `to` from the default D1 vocabulary. It also removes `not` from the full corpus, so D5 loses the negation that makes the lecture review negative, even though the notes are described positively.

### Task 5 – Vocabulary limits

The full vocabulary has 32 features, whereas `min_df=2` retains seven terms that appear in at least two documents. `max_features=5` retains `free`, `lecture`, `now`, `the` and `win` in this run, favoring high total frequencies. Several terms tie in frequency, so the exact selection at the cutoff can depend on the library implementation.

### Task 6 – N-grams

The bigram `not good` preserves the negative opinion about the lecture better than the unigram `good`, while `were great` helps describe the positive opinion about the notes. Across the full corpus the vocabulary grows from 32 unigrams to 67 unigram/bigram features and 98 unigram/bigram/trigram features. That is more than three times the initial size, even in this small corpus.

### Task 7 – TF-IDF

| D2 word | Count | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.389141 |
| cash | 1 | 0.389141 |
| entry | 1 | 0.389141 |
| free | 1 | 0.313957 |
| now | 1 | 0.313957 |
| prizes | 1 | 0.389141 |
| today | 1 | 0.313957 |
| win | 1 | 0.313957 |

Although every D2 term appears once, `cash` has a higher weight than `today` because cash appears in fewer documents. In D1, `win` has the highest weight (0.573889) because it occurs twice, which outweighs its lower IDF relative to terms appearing in just one document.

### Task 8 – Textbook calculation

Using length-normalized TF and natural logarithms, N = 5 and D2 contains 8 analyzed tokens.

**cash:**

- TF = 1/8 = 0.125000
- IDF = ln(5/1) = 1.609438
- TF-IDF = 0.125000 × 1.609438 = 0.201180
- With raw-count TF, the weight is 1 × 1.609438 = 1.609438.

**today:**

- TF = 1/8 = 0.125000
- IDF = ln(5/2) = 0.916291
- TF-IDF = 0.125000 × 0.916291 = 0.114536
- With raw-count TF, the weight is 1 × 0.916291 = 0.916291.

Both textbook conventions rank cash higher than today, matching scikit-learn. The numerical weights differ because scikit-learn uses smoothing, an added IDF constant and L2 normalization.

### Task 9 – Sparse matrix

The count matrix has 160 cells, of which 39 are nonzero and 121 are zero. The script prints 75.62% zeros (rounded from 75.625%). This demonstrates why sparse storage is useful.

### Task 10 – Fit and transform

The unknown words in N1 are `prize` and `your`, and in N2 they are `is`, `moved` and `tomorrow`. `prize` is unknown because the vectorizer treats it as a distinct spelling from `prizes`. Stemming or lemmatization applied consistently before vectorization could map both to the same form.

### Task 11 – Data leakage

The training-only vocabulary excludes `is`, `moved`, `prize`, `tomorrow` and `your`, but fitting on training and test text adds those terms. This exposes information from the test set before evaluation. Even though this demonstration does not train a classifier or measure accuracy, it shows how the feature space can be contaminated.

### Task 12 – No meaning

The cosine similarity between R1 and R2 is 0.75, exactly the same as between R1 and R3. Each pair shares three words, and the different adjectives occupy separate columns. Therefore the model cannot recognize that `tasty` and `delicious` express similar opinions while `terrible` expresses the opposite.

### Recommended settings for a spam detector

I would start with lowercase TF-IDF word unigrams and bigrams, and compare the results with a simple count-based baseline such as Multinomial Naive Bayes. The experiment shows that words such as `cash` can receive greater weight than more widespread terms such as `today`, while bigrams can preserve useful phrases. I would avoid blindly removing English stopwords because the experiment removed `not` and `now`, which may carry useful information. On a large corpus, I would tune a modest `min_df` and a reasonable feature limit using training-only cross-validation, rather than restricting the model to five words and losing signals like `cash` and `prizes`. Character n-grams may help with altered spellings, but I would avoid unnecessarily large n-gram ranges due to the observed vocabulary growth. These are proposed settings, not measured spam-classification results, because the supplied corpus is too small to establish a reliable classifier evaluation.

## References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press, especially chapters 2 and 6. https://nlp.stanford.edu/IR-book/
2. Scikit-learn developers. *Feature extraction: text feature extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html
3. SciPy developers. *Compressed sparse row matrix (csr_matrix)*. https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
4. Scikit-learn developers. *CountVectorizer API reference*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
5. Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526
6. Scikit-learn developers. *TfidfVectorizer API reference*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
7. Scikit-learn developers. *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html
8. *JCT at SemEval-2023 Tasks 12 A and 12B: Sentiment Analysis for Tweets Written in Low-resource African Languages using Various Machine Learning and Deep Learning Methods, Resampling, and HyperParameter Tuning*. (2023). Proceedings of the 17th International Workshop on Semantic Evaluation, 365–378. https://aclanthology.org/2023.semeval-1.49/ (Tables 4 and 5).

Research and code assistance: AI was used to help locate sources, draft explanations and implement the experiments. The practical numerical results were generated by executing the accompanying Python script.
