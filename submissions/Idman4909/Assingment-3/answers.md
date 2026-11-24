# Assignment 3 – Vectorization

**Student:** Idman Ahmed Ali  
**Course:** NLP for Machine Learning Bootcamp  


# Part A – Research

## 1. Text vectorization and one-hot encoding

Text vectorization is the process of converting words or documents into numerical feature representations that an algorithm can process. Most traditional machine-learning algorithms expect a fixed number of input features: each position in a vector must have the same meaning for every example. A shared vocabulary provides those positions, so a document can be represented as a vector of the same length as every other document.

In one-hot encoding, each vocabulary word receives its own position. A word is represented by a vector containing one `1` at its position and `0` everywhere else. For example, with the vocabulary `[cash, class, free, lunch, notes, win]`, `free` is `[0, 0, 1, 0, 0, 0]`. One-hot encoding represents an individual word, not a complete document by itself. A sentence such as “win free cash” therefore produces three separate word vectors; a document-level representation needs an additional method to combine or arrange them. In a real email-filtering system, one-hot encoding can identify the word “free”, but it does not by itself summarize all the words in the email in one fixed-size document vector.

## 2. Bag of Words and the document-term matrix

Bag of Words (BoW) represents a document by the words it contains, usually recording each word's count. A document-term matrix organizes these values: rows represent documents, columns represent vocabulary terms, and each cell gives the count of a column word in a row document. It is called a “bag” because the representation ignores the original sequence of words; it records which terms occur and how often, rather than their positions.

Ignoring order may be acceptable for topic classification, such as identifying whether an article discusses sports or technology. It can be harmful for sentiment or meaning when order changes the interpretation. For example, “not good” and “good” share the word “good”, but the negation changes the sentiment. A unigram BoW representation cannot preserve that relationship.

## 3. Vocabulary size and sparsity

Vocabulary size is the number of distinct features learned from the training corpus. A document-term matrix is sparse because each document normally uses only a small fraction of all terms in the complete corpus. As the number of documents and unique words grows, the matrix can become extremely wide, while most document-word combinations remain zero. Sparse formats such as CSR store non-zero values together with their column indices and row boundaries instead of storing every zero.

`min_df` removes terms that occur in fewer than a specified number of documents; `max_df` removes terms occurring in too many documents; and `max_features` keeps only a limited number of terms, commonly selected by corpus frequency. Lowercasing merges forms such as `Free` and `free`. Stop-word removal excludes selected common words, while stemming can merge related forms such as `connect`, `connected`, and `connecting` into a shared stem. These choices reduce or alter the vocabulary, but should be selected with the task and language in mind.

## 4. N-grams

An n-gram is a consecutive sequence of n tokens. Unigrams contain one word, bigrams contain two adjacent words, and trigrams contain three. For the phrase “not very good”, the unigrams include `not`, `very`, and `good`; bigrams include `not very` and `very good`; and the trigram is `not very good`. The bigram `not good` can preserve a negation cue that a unigram model treats as separate words.

Larger n-grams create more possible features and often produce a much larger, sparser vocabulary. They can capture local word order and useful phrases, but rare phrases may appear only once and generalize poorly. They also increase memory use and computation, so n-gram range should be chosen using validation rather than assumed to improve every model.

## 5. TF-IDF and inverse document frequency

TF-IDF means term frequency multiplied by inverse document frequency. Term frequency measures how often a term occurs in a document. Inverse document frequency reduces the weight of terms that occur in many documents and increases the relative importance of terms that occur in fewer documents. A common textbook form is:

\[
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\times\log\left(\frac{N}{\operatorname{df}(t)}\right)
\]

Here, \(N\) is the number of documents and \(\operatorname{df}(t)\) is the number of documents containing term \(t\). The historical development of IDF is associated with Karen Spärck Jones's work on term specificity in information retrieval (1972); later vector-space research, including Salton, Wong and Yang (1975), helped establish term-weighted document representations.

Scikit-learn's `TfidfVectorizer` uses a smoothed IDF by default:

\[
\operatorname{idf}(t)=\log\left(\frac{1+N}{1+\operatorname{df}(t)}\right)+1
\]

It also applies L2 normalization to each document vector by default. Its default term frequency is the raw count; `sublinear_tf=True` changes this to \(1+\log(tf)\). These implementation details mean its numerical values need not equal the unnormalized textbook calculation. TF-IDF values are weights, not probabilities: they are not constrained to represent mutually exclusive outcomes or sum to one as a probability distribution. L2 normalization makes the vector's Euclidean length one, not the sum of its values one.

## 6. Fit, transform and data leakage

`fit` learns the vocabulary and, for TF-IDF, document-frequency statistics from the supplied training documents. `transform` applies those learned rules to new documents without learning new vocabulary. `fit_transform` learns the representation and transforms the same input, usually more efficiently. A word not present in the fitted vocabulary is out of vocabulary (OOV) and does not receive a feature in the resulting vector; it is effectively ignored by a standard word vectorizer.

Data leakage occurs when information from the test set is used while learning preprocessing parameters, including the vocabulary or IDF weights. This can make evaluation appear better or otherwise differ from a real deployment, because the model has indirectly used information from data that should have been unseen. The correct order is to split data first, fit the vectorizer only on training text, transform training text, transform validation/test text using that fitted vectorizer, and then evaluate. Any learned preprocessing, feature selection, or scaling must follow the same rule.

## 7. Case study: term-frequency features and transformers for African-language sentiment

Aryal, Prioleau, Shah and Acharya (2023) study sentiment classification across African languages using datasets released for the AfriSenti-SemEval shared task. They evaluate Delta TF-IDF features with conventional machine-learning models and compare their approach with transformer-based counterparts. Their paper reports that Delta TF-IDF can be a promising approach in low-resource African-language sentiment analysis while requiring substantially less data than transformer alternatives.

The study is relevant because it shows that count-derived lexical features remain useful when annotated data and computational resources are limited. However, TF-IDF still represents texts through weighted lexical features: it does not inherently understand that different words can express similar meanings, nor does a unigram representation reliably understand negation, context, sarcasm, or word order. Transformer representations can use contextual information and semantic relationships, although they generally require more resources and suitable language data. The paper therefore illustrates a practical trade-off rather than proving that one representation is best for every African language or task.

# Part B – Practical

The Python script is `vectorization_practice.py`. Its complete output is saved in `output.txt`. The tables and numerical values below should be checked against that generated output after running the script in the submission environment.

## Task 2 – Document-term matrix

The matrix below is copied from the executed script. Columns are in the exact alphabetical feature order returned by `CountVectorizer`.

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |


## Task 7 – D2 counts and TF-IDF

The following values are generated by the script. D2 contains each term once, but weights differ because document frequencies differ; the default scikit-learn weights also include smoothed IDF and L2 normalization.

| Word | Count in D2 | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.389141 |
| cash | 1 | 0.389141 |
| entry | 1 | 0.389141 |
| free | 1 | 0.313957 |
| now | 1 | 0.313957 |
| prizes | 1 | 0.389141 |
| today | 1 | 0.313957 |
| win | 1 | 0.313957 |


## Task 8 – Hand calculations

For D2, raw TF is 1 for both terms. `cash` occurs in one document and `today` occurs in two documents, so their textbook values are:

| Term | TF | df | IDF = ln(5/df) | TF-IDF |
|---|---:|---:|---:|---:|
| cash | 1 | 1 | 1.609438 | 1.609438 |
| today | 1 | 2 | 0.916291 | 0.916291 |

`cash` ranks above `today` in both textbook TF-IDF and scikit-learn's D2 weights. Scikit-learn's values have a different scale because it uses smoothed IDF and L2 normalization.

## Task 1 – One-hot encoding

Each vocabulary item is represented by a six-element vector with one active position. The phrase `win free cash` produces three vectors because it contains three tokens; one-hot encoding creates one vector per word rather than automatically combining a sentence into one document vector. The exact vectors are printed in Task 1.

## Task 2 – Bag of Words

The script learns one shared vocabulary from D1–D5 and counts each term in every document. D1 has a count of two for `win` because that word appears twice in the first document. The two test phrases `win free phone` and `phone free win` have identical BoW vectors because the representation ignores word order.

## Task 4 – Preprocessing

The default vectorizer lowercases text, so uppercase and lowercase versions of a word are merged; `lowercase=False` keeps case distinctions. The default token pattern ignores one-character tokens and punctuation, which is why `a` and `!` do not become vocabulary entries. English stop-word removal removes common listed words such as `to` and also removes `not` in scikit-learn’s built-in English list. Therefore, the stop-word-filtered vocabulary loses the negation cue in D5, which can be harmful for sentiment tasks.

## Task 5 – Vocabulary limits

`min_df=2` keeps only terms that occur in at least two documents, removing one-document terms. `max_features=5` keeps only five terms with the highest corpus term frequencies. The script prints the actual vocabularies and sizes, which show how these settings reduce the feature space.

## Task 6 – N-grams

Unigrams represent individual words, while bigrams add adjacent two-word sequences. In D5, the bigram `not good` is especially useful because it preserves a negation phrase; the script prints the complete D5 n-gram vocabulary. Increasing the n-gram range from unigrams to bigrams and trigrams increases the number of candidate features, as shown by the three corpus vocabulary sizes.

## Task 7 – TF-IDF

TF-IDF reduces the influence of words shared by many documents and gives more weight to terms that are more specific to a document. D2 has a count of one for each term, but terms such as `cash` can receive higher weights than terms appearing in several documents. The script prints the complete matrix and the D1/D2 word-level values rounded or formatted for inspection.

## Task 8 – Hand calculation

The hand calculation uses the textbook formula \(tfidf=tf\times\ln(N/df)\), with \(N=5\). `cash` occurs in fewer documents than `today`, so it receives a larger IDF. Scikit-learn uses smoothed IDF and L2 normalization by default, so its numerical values differ from the textbook scores even when the relative ranking is the same.

## Task 10 – Fit and transform

The vectorizer is fitted only on D1–D4, then used to transform N1 and N2. Words absent from the training vocabulary are OOV and have no corresponding columns in the transformed vectors. `prize` is unknown because the training corpus contains `prizes`, a different token; stemming or lemmatization from preprocessing can help normalize related word forms.

## Task 11 – Data leakage

The first vectorizer learns vocabulary only from D1–D4, while the second also sees N1 and N2. Any terms present only in the test texts and added to the second vocabulary are leakage for a proper evaluation workflow. The correct procedure is to split first, fit on training documents only, and transform test documents without refitting.

## Task 12 – Why BoW does not capture meaning

The three review vectors share the words `the`, `food`, and `was`, while `tasty`, `delicious`, and `terrible` are separate features. Consequently, the cosine similarity between R1 and R2 is not automatically higher than the similarity between R1 and R3: the vectorizer does not know that `tasty` and `delicious` are synonyms or that `terrible` has an opposite sentiment. The script prints the exact cosine similarity matrix and comparison.

## Recommended vectorizer settings for a spam detector

For a spam detector, I would begin with word-level TF-IDF and compare it against raw or binary counts using a validation set. I would test unigrams together with bigrams because phrases such as “claim prize”, “free entry”, and “reply now” may be more informative than isolated words. I would be cautious about removing stopwords: words such as `not` can change meaning, and short function words may sometimes be useful. I would also avoid aggressive vocabulary limits until validation shows they improve performance, because rare words, numbers, or unusual promotional terms can be useful spam indicators. All vocabulary learning and TF-IDF fitting must use training data only.

# References

1. Jones, K. S. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526
2. Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM, 18*(11), 613–620. https://doi.org/10.1145/361219.361220
3. Aryal, S. K., Prioleau, H., Shah, U., & Acharya, S. (2023). Term Frequency Features vs Transformers: A Comparison for Sentiment Classification of African Languages. *Journal of Computing Sciences in Colleges*. https://www.ccsc.org/publications/journals/EA2023.pdf
4. scikit-learn developers. (n.d.). *Feature extraction: Text feature extraction and vectorization*. https://scikit-learn.org/stable/modules/feature_extraction.html
5. scikit-learn developers. (n.d.). *TfidfVectorizer API documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
