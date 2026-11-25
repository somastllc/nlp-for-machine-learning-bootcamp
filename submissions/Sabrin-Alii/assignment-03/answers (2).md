# Assignment 3 - Text Vectorization

## Part A - Research

### 1. What is text vectorization?
Text vectorization is the process of converting written documents into numerical feature representations. Many traditional machine-learning algorithms, including Logistic Regression, operate on numerical feature vectors rather than raw sentences. A fixed vocabulary gives every document the same feature positions, so each document can be represented in a consistent feature space.

In one-hot encoding, each word is represented by a vector with one `1` at the position assigned to that word and `0` at all other positions. For example, using the vocabulary `[apple, clinic, doctor, nurse]`, the word `doctor` becomes `[0, 0, 1, 0]`. A sentence becomes a list of word vectors, one for each token. The number of vectors changes with sentence length, so this does not by itself give a traditional document classifier one fixed-length vector for a whole document. Bag of Words addresses this by summarizing term counts across the document.

A real-life example is a hospital feedback classifier. Each patient comment can be converted into a vector based on a shared vocabulary, and a model can classify the feedback into categories such as appointment, staff, facilities, or billing.

### 2. Bag of Words and the document-term matrix
Bag of Words (BoW) represents a document by counting occurrences of vocabulary terms. It is called a “bag” because it keeps terms and counts but does not preserve the original word order. In a document-term matrix (DTM), rows represent documents, columns represent unique vocabulary terms, and cells contain the count of a term in a document.

For a product-review classifier, ignoring word order may be acceptable when identifying a broad topic from terms such as “battery”, “screen”, or “delivery”. However, it can be harmful when order changes who did what. “The manager approved the request” and “The request approved the manager” contain the same words but describe different relationships. Basic BoW cannot distinguish them from word counts alone.

### 3. Vocabulary size and sparsity
A token is one occurrence of a word. The vocabulary is the set of unique terms retained by the vectorizer. Vocabulary size is the number of unique retained terms and determines the number of DTM columns. Repeated appearances increase a term's count but do not create additional vocabulary columns.

A matrix can become large because a collection contains many distinct terms, while each document uses only a small part of the vocabulary. Most cells are therefore zero; this is called sparsity. Compressed sparse row (CSR) formats store non-zero values, their column positions, and row boundary information rather than every zero. Scikit-learn's `CountVectorizer` returns a sparse count representation.

`min_df` removes terms occurring in fewer than a chosen minimum number or proportion of documents. `max_df` removes terms occurring in more than a chosen maximum number or proportion of documents. `max_features` limits the vocabulary to the most frequent terms. Lowercasing combines case variants. Stopword removal removes a selected list of common terms, but the list should be checked because some common terms carry meaning. Stemming reduces related word forms to a shared stem, while lemmatization maps inflected forms to a dictionary form. These methods can reduce separate vocabulary entries, but careless normalization can remove useful distinctions.

### 4. N-grams
An n-gram is a sequence of n neighboring tokens. A unigram contains one token, a bigram contains two consecutive tokens, and a trigram contains three. Bigrams and trigrams preserve a small amount of local word order. For example, “barely useful” communicates a different opinion from “very useful”; keeping neighboring words together can preserve this phrase-level clue.

Larger n-gram ranges increase vocabulary size because each new sequence can become a separate feature. This increases memory use and sparsity and may add many rare sequences. The range should be selected for the task and checked using validation data.

### 5. TF-IDF
TF-IDF means Term Frequency-Inverse Document Frequency. It assigns a weight based on how often a term occurs in one document and how widely that term occurs across the whole collection. The inverse document frequency idea is strongly associated with Karen Spärck Jones's 1972 paper on statistical term specificity in information retrieval. Her work argued that a term's collection frequency can help estimate how useful or discriminating it is for retrieval.

Textbook formula used in this assignment:

- `TF(t,d) = count(t in d) / total number of tokens in d`
- `IDF(t) = ln(N / df(t))`
- `TF-IDF(t,d) = TF(t,d) * IDF(t)`

Here, `N` is the total number of documents and `df(t)` is the number of documents containing the term. If two terms occur equally often in a document, the term appearing in fewer documents receives the larger IDF and TF-IDF weight.

The default scikit-learn `TfidfVectorizer` uses raw term counts as the TF component, rather than dividing each count by document length. Its default IDF is smoothed: `IDF(t) = ln((1 + N)/(1 + df(t))) + 1`. It also applies L2 normalization to each document vector. Therefore, its values differ from manual calculations using the textbook formula. TF-IDF values are feature weights, not probabilities.

### 6. fit, transform, fit_transform, and data leakage
- `fit()` learns the vocabulary from the supplied documents. For TF-IDF, it also learns document-frequency information used for IDF.
- `transform()` converts documents using the learned vocabulary and statistics. A word outside the fitted vocabulary is out-of-vocabulary (OOV) and does not create a new column.
- `fit_transform()` learns the representation and transforms the same input data in one step.

Data leakage occurs when information from data reserved for evaluation is used while preparing the training representation. Fitting a vectorizer on all documents before splitting allows vocabulary and IDF statistics to include information from the test set, making the evaluation procedure unreliable and potentially optimistic. The correct order is to split first, fit preprocessing and vectorization on training text only, then transform both training and test text using the fitted objects.

### 7. Case study: Tigrinya news classification
Fesseha et al. (2021) studied text classification for Tigrinya, a lower-resource language spoken in Eritrea and northern Ethiopia. Their manually categorized news dataset contained 30,000 documents in six topics: sport, agriculture, politics, religion, education, and health. They used 24,000 articles for training and 6,000 for testing. They also collected an unlabelled corpus containing more than six million words to support word-embedding work.

For traditional machine learning, the researchers evaluated Support Vector Machines, Naive Bayes, Decision Trees, and Random Forest with Bag-of-Words features represented using unigram TF-IDF vectors. They also evaluated CNN approaches using word representations, including Word2Vec with CBOW and Skip-gram and FastText. The paper reports that the CNN using Word2Vec with CBOW achieved 93.41% accuracy and an F1 score of 0.9274. It also reports that Random Forest had the best validation accuracy among the traditional models.

The study compares frequency-based representations with embedding-based representations for an African language. It discusses that frequency-based approaches focus on word occurrences and do not capture contextual structure or word meaning in the same way as embeddings. Word2Vec improved the reported classification performance by learning relationships among words. These results are specific to the Tigrinya dataset and should not be presented as measured results for Somali. The paper compares unigram TF-IDF/BoW with embedding-based CNN methods; it does not compare every possible n-gram range.

### References
1. Fesseha, A., Xiong, S., Emiru, E. D., Diallo, M., & Dahou, A. (2021). Text Classification Based on Convolutional Neural Networks and Word Embedding for Low-Resource Languages: Tigrinya. *Information, 12*(2), 52. https://doi.org/10.3390/info12020052
2. Spärck Jones, K. (1972). A Statistical Interpretation of Term Specificity and Its Application in Retrieval. *Journal of Documentation, 28*(1), 11-21. https://doi.org/10.1108/eb026526
3. Scikit-learn Developers. *CountVectorizer documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
4. Scikit-learn Developers. *TfidfVectorizer documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5. Scikit-learn Developers. *Feature extraction: text feature extraction and TF-IDF term weighting*. https://scikit-learn.org/stable/modules/feature_extraction.html
6. Scikit-learn Developers. *Common pitfalls and recommended practices: data leakage*. https://scikit-learn.org/stable/common_pitfalls.html

## Part B - Practical report

The script `vectorization_practice.py` prints the results for all twelve tasks. Its full output is saved in `output.txt`.

### Task 1 - One-hot encoding
The vocabulary is `[cash, class, free, lunch, notes, win]`, so every one-hot vector has six positions. `free cash` produces two word vectors, while `win free cash` produces three. The vector length stays six, while the number of vectors follows sentence length.

### Task 2 - Bag of Words
The DTM shape is 5 documents by 32 terms. The D1 value for `win` is 2 because `Win` and `WIN` are lowercased to the same token. `win free phone` and `phone free win` produce identical count vectors because basic BoW ignores order. The complete DTM is printed in `output.txt` and shown in the Practical PDF.

### Task 3 - Tokens and vocabulary
Total tokens: **41**. Vocabulary size: **32**. Words appearing in more than one document: **free, lecture, notes, now, the, today, win**.

### Task 4 - Vectorizer preprocessing
On D1, the default vocabulary is `claim, free, now, phone, reply, to, win`; with `lowercase=False`, it is `FREE, Reply, WIN, Win, claim, now, phone, to`; with `stop_words="english"`, it is `claim, free, phone, reply, win`. The one-letter `a` is excluded by the default token pattern, and `!` is punctuation. Lowercasing merges `Win` and `WIN`; without lowercasing, they remain separate. On D1-D5, the English stopword list removes `not`, which removes the negation cue from D5.

### Task 5 - Vocabulary limits
- Full vocabulary: **32** terms.
- `min_df=2`: **7** terms - `free, lecture, notes, now, the, today, win`.
- `max_features=5`: **5** terms - `free, lecture, now, the, win`.

`min_df=2` keeps words found in at least two documents. `max_features=5` keeps five terms with the highest total term frequency.

### Task 6 - N-grams
For D5, `(1,1)` gives 9 unigram features and `(1,2)` gives 18 features, including the bigram `not good`. For D1-D5, the vocabulary sizes are 32 for `(1,1)`, 67 for `(1,2)`, and 98 for `(1,3)`. Larger n-gram ranges increase the number of features.

### Task 7 - TF-IDF
The TF-IDF matrix (rounded to two decimals) and all D1 and D2 term counts and weights are printed in `output.txt`. Every D2 term occurs once, but weights differ because the terms have different document frequencies. In D1, `win` has count 2 and the highest D1 TF-IDF weight, **0.573889**.

D2 term comparison:

| Term | Count | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.389141 |
| cash | 1 | 0.389141 |
| entry | 1 | 0.389141 |
| free | 1 | 0.313957 |
| now | 1 | 0.313957 |
| prizes | 1 | 0.389141 |
| today | 1 | 0.313957 |
| win | 1 | 0.313957 |

### Task 8 - Hand TF-IDF for D2
D2 has 8 tokens and the collection has `N=5` documents.

| Term | Count | TF | df | IDF | TF-IDF |
|---|---:|---:|---:|---:|---:|
| cash | 1 | 1/8 = 0.125000 | 1 | ln(5/1) = 1.609438 | 0.201180 |
| today | 1 | 1/8 = 0.125000 | 2 | ln(5/2) = 0.916291 | 0.114536 |

The hand calculation and scikit-learn both rank `cash` above `today`, although their exact weights differ because scikit-learn uses smoothing and L2 normalization.

### Task 9 - Sparse matrix
The matrix has 5 x 32 = **160** cells, with **39** non-zero cells and **121** zero cells. The percentage of zeros is **75.625%**.

### Task 10 - Fit and transform
After fitting on D1-D4, the OOV terms are:
- N1: `prize`, `your`.
- N2: `is`, `moved`, `tomorrow`.

The training data contains `prizes`, while N1 contains `prize`. CountVectorizer does not automatically normalize inflected word forms. Stemming or lemmatization from preprocessing can help normalize them when configured before vectorization.

### Task 11 - Data leakage
Fitting only on D1-D4 gives the training vocabulary. Fitting incorrectly on D1-D4 together with N1 and N2 introduces these test words: **is, moved, prize, tomorrow, your**. The correct procedure is to fit on training documents only and transform the test documents with that fitted vectorizer.

### Task 12 - No meaning
The review vectors and cosine similarity matrix are printed by the script. R1 (`tasty`) is not more similar to R2 (`delicious`) than to R3 (`terrible`) under BoW cosine similarity: both values are **0.750**. The reviews share `the`, `food`, and `was`, while their opinion words occupy different columns. BoW does not understand that `tasty` and `delicious` are semantically related.

### Spam detector recommendation
For a spam detector, compare CountVectorizer and TfidfVectorizer on the same training and validation split. Begin with unigrams and test a limited unigram-plus-bigram setting, because short combinations may provide useful clues. Avoid removing stopwords blindly: the English list removes `not`, which can change meaning, and it also removes `now`, which may be a useful spam clue. Use `min_df`, `max_df`, and `max_features` carefully so useful rare terms are not removed. Fit the vectorizer on training text only.
