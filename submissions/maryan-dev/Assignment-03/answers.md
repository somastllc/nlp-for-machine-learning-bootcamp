# Text Vectorization for Machine Learning: A Research Report

**Part A: Research.
---


## 1. Text vectorization and one-hot encoding

### 1.1 Definition

Text vectorization is the process of converting raw text into numbers that a machine learning algorithm can use. Most models (logistic regression, SVMs, random forests, neural networks) expect a matrix **X** of shape *(n_samples × n_features)*. Every row must have the same length, and column *j* must mean the same thing in every row.

### 1.2 Why every document needs the same number of features

Documents have different lengths. A tweet may have 12 words and a news article 800. A model learns one weight per feature column. If row 1 had 12 values and row 2 had 800, "weight number 5" would refer to different things in different rows, and the model could not learn anything consistent.

Vectorization solves this by describing every document with the same set of features, usually one feature per vocabulary word.

### 1.3 One-hot encoding of words

Take a vocabulary of *V* words. Each word is represented by a vector of length *V* that has a single 1 at that word's position and 0 everywhere else.

**Real-life example (phone reviews).** Vocabulary = {battery, screen, great, terrible, phone}.

| Word     | battery | screen | great | terrible | phone |
|----------|---------|--------|-------|----------|-------|
| battery  | 1       | 0      | 0     | 0        | 0     |
| great    | 0       | 0      | 1     | 0        | 0     |
| terrible | 0       | 0      | 0     | 1        | 0     |

### 1.4 Why one-hot encoding is not enough for a document

1. **It describes words, not documents.** The review "great phone, terrible battery" becomes four one-hot vectors (a 4×5 block). A longer review gives a taller block, so the shape still changes with document length. To get one fixed-length row you must aggregate, for example by summing the vectors. That step is exactly what Bag of Words does.
2. **Dimension equals vocabulary size.** Real vocabularies contain tens or hundreds of thousands of words, so each vector is almost all zeros.
3. **No notion of similarity.** The dot product of any two different one-hot vectors is 0. "Great" is as far from "excellent" as it is from "terrible".
4. **Order and counts are not captured** unless extra machinery is added.

---

## 2. Bag of Words and the document-term matrix

### 2.1 Structure

Bag of Words (BoW) represents each document by how often each vocabulary word occurs in it.

- **Rows** are documents.
- **Columns** are vocabulary terms (learned from the corpus).
- **Cells** hold a count of that term in that document. They can also hold a 0/1 indicator or a TF-IDF weight.

### 2.2 Example corpus

- D1 = "the teacher called the student"
- D2 = "the student called the teacher"
- D3 = "the student passed the exam"

Vocabulary (alphabetical): called, exam, passed, student, teacher, the.

| | called | exam | passed | student | teacher | the |
|----|--------|------|--------|---------|---------|-----|
| D1 | 1      | 0    | 0      | 1       | 1       | 2   |
| D2 | 1      | 0    | 0      | 1       | 1       | 2   |
| D3 | 0      | 1    | 1      | 1       | 0       | 2   |

D1 and D2 have identical rows even though their meanings differ. In D1 the teacher called the student, and in D2 the student called the teacher. The word "the" is the most frequent term but carries no information, which motivates stopword removal and IDF weighting (Section 5).

### 2.3 Why is it called a "bag"?

Imagine tipping all the words of a document into a bag and shaking it. You can still count what is inside (a multiset), but the order is gone. The idea follows the distributional view of language (Harris, 1954).

### 2.4 When losing word order does little harm, and when it hurts

- **Little harm: topic classification and spam detection.** Words like "exam", "passed" and "student" identify an education article whatever their order. The Swahili news task in Section 7 is also of this type.
- **Real problems: sentiment with negation and role-dependent tasks.** "The film was not good" and "the film was good" share most words. "The teacher called the student" and "the student called the teacher" have identical bags. Relation extraction and question answering need this information.

---

## 3. Vocabulary size and sparsity

### 3.1 Why matrices become very large

Word frequencies follow Zipf's law: a few words are extremely frequent, and there is a long tail of rare words. By Heaps' law, vocabulary keeps growing as more documents are added, because new names, typos, inflected forms and numbers keep appearing. Morphologically rich languages, with prefixes, suffixes and compounds, make this worse. The Somali study by Mohamed et al. (2025) points to the morphological complexity of Somali as a challenge for modelling.

### 3.2 Why they are mostly zeros

A single document uses only a tiny fraction of the whole vocabulary.

**Worked example.** 10,000 documents × 50,000 terms = 500 million cells. As a dense float64 array that is about 4 GB. If each document has about 100 distinct words, only about 1 million cells are non-zero, which is 0.2% of the matrix.

### 3.3 Efficient storage: sparse matrices

Sparse formats store only the non-zero entries. In SciPy:

- **CSR (compressed sparse row)** stores three arrays: `data` (the non-zero values), `indices` (their column numbers) and `indptr` (where each row starts in those arrays). Memory is roughly proportional to the number of non-zeros, not to rows × columns. scikit-learn's vectorizers return CSR matrices.
- **COO** stores (row, column, value) triples and is convenient for building a matrix.
- **CSC** is the column-oriented counterpart of CSR.

### 3.4 How settings change the vocabulary

| Setting | Effect | Trade-off |
|---|---|---|
| `min_df` | Removes terms that appear in fewer than *k* documents (integer count) or below a fraction of documents (float). | Removes typos and rare names and shrinks the vocabulary a lot, but may remove rare useful words. |
| `max_df` | Removes terms that appear in more than a given number or fraction of documents. | Works as a corpus-specific stopword filter. Too low a threshold removes useful words. |
| `max_features` | Keeps only the top-*k* terms by total frequency in the corpus. | A hard cap on dimensionality. It is crude and ignores usefulness for the label. |
| `lowercase` | "Dawa" and "dawa" become one term. | Smaller vocabulary, but may merge words that differ in meaning (e.g. names). |
| `stop_words` | Removes very frequent function words. | Shrinks the vocabulary but can remove words such as "not" that matter for negation. The scikit-learn documentation warns that its built-in English list has known issues. |
| Stemming / lemmatization | Merges inflected forms ("walking", "walked" become "walk"). | Fewer and denser features. It needs a language-specific tool and is not built into scikit-learn, so a custom tokenizer is passed in. |

---

## 4. N-grams

### 4.1 Definition

An n-gram is a sequence of *n* consecutive tokens. For "the movie was not good":

- **Unigrams:** the, movie, was, not, good
- **Bigrams:** the movie, movie was, was not, **not good**
- **Trigrams:** the movie was, movie was not, was not good

In scikit-learn this is controlled by `ngram_range=(1, 2)`. N-grams keep local word order inside a window of *n* words, but not the global order of the document.

### 4.2 Example where a bigram changes the meaning

With unigrams only, "good" and "not good" both contain the feature "good", and "not" is a separate column, so the model cannot see that "not" modifies "good". The bigram "not good" is a feature of its own, so a linear model can give it a negative sentiment weight. Likewise, the bigram "new york" is not the same as the two unigrams "new" and "york".

### 4.3 Cost of larger n-grams

- **Vocabulary explosion.** The number of possible n-grams grows roughly like *V* to the power *n*, and the number actually observed grows quickly. Zhang et al. (2015) had to truncate: their n-gram baselines kept the 500,000 most frequent n-grams (up to 5-grams), compared with 50,000 words for plain BoW.
- **More sparsity.** Most trigrams occur only once or twice, so each feature has little evidence behind it.
- **Overfitting and memory.** Models can memorise rare n-grams that do not generalise, and the matrices need much more memory.
- **Diminishing returns.** Wang & Manning (2012) found that bigrams often help, but going further usually adds little for classification.

---

## 5. TF-IDF

### 5.1 Origin of inverse document frequency

The idea comes from Karen Spärck Jones (1972), "A statistical interpretation of term specificity and its application in retrieval". Her argument was that a term's usefulness for retrieval depends on its specificity, meaning how few documents contain it. Salton & Buckley (1988) systematised how to combine term frequency with this weight, and Robertson (2004) studied the theoretical justifications for IDF.

### 5.2 Textbook formula (Manning et al., 2008)

- tf(t, d) = how often term *t* occurs in document *d* (raw count, or a normalised or log-scaled version)
- idf(t) = log( N / df(t) ), where N is the number of documents and df(t) is the number of documents containing *t*
- **tf-idf(t, d) = tf(t, d) × idf(t)**

### 5.3 Why rarer words get higher weights

Suppose N = 1000 documents.

| df(t) | idf = ln(N / df) | Interpretation |
|---|---|---|
| 1000 (in every document) | 0.00 | A word like "the" tells us nothing. |
| 100 | 2.30 | Moderately informative. |
| 1 | 6.91 | Very specific to one document. |

The logarithm dampens the effect, so a word in one document is not treated as a thousand times more important than a word in all documents.

### 5.4 Worked example on the Section 2 corpus (N = 3)

| Term | df | Textbook idf = ln(3/df) | scikit-learn idf = ln((1+3)/(1+df)) + 1 |
|---|---|---|---|
| the | 3 | 0.000 | 1.000 |
| student | 3 | 0.000 | 1.000 |
| called | 2 | 0.405 | 1.288 |
| teacher | 2 | 0.405 | 1.288 |
| exam | 1 | 1.099 | 1.693 |
| passed | 1 | 1.099 | 1.693 |

"Exam" and "passed" appear in only one document, so they receive the highest weights and define D3. "The" and "student" appear everywhere, so the textbook formula gives them zero weight.

### 5.5 How scikit-learn's TfidfVectorizer differs from the textbook

According to the scikit-learn documentation (TfidfTransformer):

1. **Smoothed idf (default).** idf(t) = ln[(1 + N) / (1 + df(t))] + 1. One is added to numerator and denominator, as if an extra document containing every term once had been seen. This prevents division by zero. The documentation itself says this differs from the standard textbook notation.
2. **The extra "+1" outside the log.** Terms that occur in every document get idf = 1 rather than 0, so they are not completely ignored.
3. **L2 normalisation (default).** Each row is scaled to unit Euclidean length, so documents of different lengths are comparable and cosine similarity becomes a plain dot product.
4. **Optional sublinear tf.** With `sublinear_tf=True`, tf is replaced by 1 + log(tf).

**Check.** The documentation's own example (4 documents) shows idf values 1.0, 1.2231, 1.5108 and 1.9163 for terms appearing in 4, 3, 2 and 1 documents. These match ln(5/5)+1, ln(5/4)+1, ln(5/3)+1 and ln(5/2)+1.

### 5.6 TF-IDF values are weights, not probabilities

- idf can be greater than 1, so values are not confined to [0, 1] before normalisation.
- After L2 normalisation it is the squares of the values that sum to 1, not the values themselves.
- The formula does not model any random event. A value of 0.3 does not mean "30% likely" of anything. It is a heuristic importance score.

Only plain relative term frequency (count divided by document length) behaves like an estimated word probability.

---

## 6. `fit`, `transform`, `fit_transform`, and data leakage

### 6.1 The three methods

| Method | What it does |
|---|---|
| `fit(X)` | **Learns** from the data. For `CountVectorizer`, it learns the vocabulary. For `TfidfVectorizer`, it learns the vocabulary and the idf vector. |
| `transform(X)` | **Applies** the already-learned vocabulary and idf to documents and returns the matrix. Nothing new is learned. |
| `fit_transform(X)` | Runs `fit` and then `transform` on the same data in one step. |

### 6.2 Out-of-vocabulary (OOV) words

If a new document contains a word that was not seen during `fit`, `transform` has no column for it, so the word is silently ignored. A document made only of unseen words becomes an all-zero row. This is realistic: in deployment, new words will always appear.

### 6.3 Data leakage in vectorization

Leakage means information from the test data influences the training pipeline. If you call `fit_transform` on the whole dataset and split afterwards, then the vocabulary, the idf values (which depend on document frequencies) and any `min_df`, `max_df` or `max_features` cut-offs were computed using test documents.

**Why results look better than they really are:**

- Test words are guaranteed to be in the vocabulary, so the model never faces real OOV words.
- The idf weights already reflect the test distribution.
- Any feature selection has already "seen" the test set.

My own assessment is that unsupervised leakage of this kind is often modest, but it is still a methodological error. It can become severe if label-based feature selection (for example chi-squared) is done before splitting.

### 6.4 Correct order of steps

1. Split the data into train and test sets (stratified for classification).
2. `fit` the vectorizer on the **training** texts only.
3. `transform` the training texts and the test texts with that fitted vectorizer.
4. Train the classifier on the training matrix.
5. Evaluate once on the test matrix.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
Xtr = vec.fit_transform(X_tr)   # learn vocabulary + idf from TRAIN only
Xte = vec.transform(X_te)       # only apply to TEST

clf = LogisticRegression(max_iter=1000).fit(Xtr, y_tr)
print(clf.score(Xte, y_te))
```

For cross-validation, wrap the vectorizer and classifier in a `sklearn.pipeline.Pipeline` so the vectorizer is re-fitted inside every training fold.

---

## 7. Case study: Swahili news classification

**Source.** Kastanos, A. & Martin, T. (2021). *Graph Convolutional Network for Swahili News Classification.* arXiv:2103.09325.

### 7.1 Task

The task is to classify Swahili news articles into six categories: national, sports, entertainment, economy, international and health. The authors deliberately focus on a realistic low-resource African-language setting, where few labelled examples are available and computing power is limited.

### 7.2 Data

The Swahili News Classification dataset (David, 2020) contains 23,266 labelled articles, split 8:1:1 into train, validation and test sets. The classes are very imbalanced: national news has about 8,200 training documents, while health has about 690. For that reason the authors use macro F1 as their main metric. Preprocessing included lowercasing, stopword removal and stemming with the SALAMA Swahili language manager.

### 7.3 Representations compared

All feature vectors are fed to a logistic regression classifier. The representations are:

- TF-IDF
- Counts (the unnormalised version, i.e. plain Bag of Words)
- Averaged pre-trained fastText word embeddings
- doc2vec (PV-DBOW and PV-DM)
- Text GCN, a graph network over documents and words that uses TF-IDF for document-word edges and PPMI (word co-occurrence) for word-word edges, plus a variant (Text GCN-t2v) that uses word2vec/doc2vec inputs

### 7.4 Results (test set)

| Model | Accuracy (%) | Macro F1 (%) |
|---|---|---|
| TF-IDF | 83.07 | 68.72 |
| Counts | 83.32 | 73.60 |
| Averaged fastText | 67.47 | 32.41 |
| PV-DBOW | 81.64 | 72.93 |
| PV-DM | 77.01 | 67.50 |
| Text GCN | 84.62 | 75.29 |
| Text GCN-t2v | **85.40** | **75.67** |

### 7.5 How the study relates to the main limitation of BoW and TF-IDF

The paper does **not** explicitly state that BoW and TF-IDF do not capture meaning. The link below is my interpretation of the evidence.

1. **Extra relational information helped.** Text GCN keeps TF-IDF but adds information about which words co-occur. Both GCN variants beat all traditional baselines. When the co-occurrence information was removed entirely, F1 dropped to 73.33%, from 75.29%. This suggests that relationships between words add something that independent word weights miss.
2. **Few labels hurt BoW more.** The authors found that TF-IDF and Counts degraded sharply when the share of labelled training data fell from 5% to 1%, while the graph models held up better. A plausible reading is that in BoW every word is an independent column, so related words cannot share statistical strength. Kowsari et al. (2019) describe this limitation in their survey: TF-IDF cannot represent similarity between words, because each word is a separate index.
3. **Caveat: "semantic" features are not automatically better.** Averaged fastText vectors performed worst by far (F1 32.41), and plain Counts beat TF-IDF on macro F1. Averaging embeddings can wash out topical signal, and Swahili's rich morphology and limited pre-training data may also hurt. The fair conclusion is that simple baselines are strong for topic classification, and that adding relational information gave a modest gain of about 2 F1 points over the best baseline.
4. **Note on the setup.** The paper says all baseline feature vectors were 300-dimensional. For TF-IDF and Counts that implies a heavily truncated vocabulary, which may understate their full-vocabulary performance. This is worth mentioning as a limitation of the comparison.

### 7.6 Related Somali work

Mohamed et al. (2025) built SomBERTa, a Somali transformer language model. It outperformed multilingual models such as AfriBERTa and AfroXLMR on fake-news and toxicity classification. I read the abstract and parts of the text only, and I did not verify whether the paper includes BoW or TF-IDF baselines, so it is not used as the main case study.

---

## 8. References

**Read in this session**

- Kastanos, A. & Martin, T. (2021). Graph Convolutional Network for Swahili News Classification. arXiv:2103.09325.
- scikit-learn documentation: `TfidfTransformer` and `TfidfVectorizer`. https://scikit-learn.org
- Kowsari, K., Jafari Meimandi, K., Heidarysafa, M., Mendu, S., Barnes, L. & Brown, D. (2019). Text Classification Algorithms: A Survey. arXiv:1904.08067.
- Zhang, X., Zhao, J. & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. arXiv:1509.01626.
- Mohamed, M. A. et al. (2025). Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models. arXiv:2503.18117.

**Standard references (cited from background knowledge; please verify)**

- Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21.
- Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5).
- Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
- Robertson, S. (2004). Understanding inverse document frequency: on theoretical arguments for IDF. *Journal of Documentation*, 60(5).
- Harris, Z. (1954). Distributional structure. *Word*, 10(2–3).
- Wang, S. & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *Proceedings of ACL (Short Papers).*

---

# Practical Section

This section uses the output of my Python script (scikit-learn `CountVectorizer` and `TfidfVectorizer`) on five short messages. The first two are spam-like (D1, D2) and the other three are ordinary messages (D3, D4, D5).

| ID | Text |
|---|---|
| D1 | Win a FREE phone now! Reply WIN to claim. |
| D2 | Free entry: win cash prizes today. Call now. |
| D3 | Are we still meeting for lunch today? |
| D4 | Please send me the lecture notes before class. |
| D5 | The lecture was not good, but the notes were great. |

## P1. Document-term matrix (Task 2)

Shape: 5 documents × 32 vocabulary words. Words are in alphabetical order, and each cell is a count. The word "a" in D1 does not appear because the default tokenizer ignores one-letter tokens.

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **D1** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| **D2** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **D3** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **D4** | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **D5** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

## P2. Count versus TF-IDF weight for every word in D2 (Task 7)

D2 = "Free entry: win cash prizes today. Call now." The scikit-learn idf values in the last column are my own computation, using idf = ln((1+6)/(1+df)) + 1 with 5 documents (N = 5), and they reproduce your printed weights after L2 normalisation.

| Word | Count in D2 | Documents containing it (df) | scikit-learn idf | TF-IDF weight |
|---|---|---|---|---|
| call | 1 | 1 | 2.099 | 0.389 |
| cash | 1 | 1 | 2.099 | 0.389 |
| entry | 1 | 1 | 2.099 | 0.389 |
| free | 1 | 2 | 1.693 | 0.314 |
| now | 1 | 2 | 1.693 | 0.314 |
| prizes | 1 | 1 | 2.099 | 0.389 |
| today | 1 | 2 | 1.693 | 0.314 |
| win | 1 | 2 | 1.693 | 0.314 |

Every word has the same count (1), but the weights split into two groups. Words found only in D2 (call, cash, entry, prizes) get 0.389, and words shared with another document (free, now, today, win) get 0.314.

## P3. Hand calculations (Task 8)

D2 has 8 tokens, and N = 5 documents in the corpus.

**Word "cash"**: count = 1, df = 1.

- TF = 1 / 8 = 0.125
- IDF = ln(5 / 1) = ln 5 = 1.6094
- TF-IDF = 0.125 × 1.6094 = **0.2012**

**Word "today"**: count = 1, df = 2 (D2 and D3).

- TF = 1 / 8 = 0.125
- IDF = ln(5 / 2) = ln 2.5 = 0.9163
- TF-IDF = 0.125 × 0.9163 = **0.1145**

**How scikit-learn gets its numbers** (my reconstruction, which matches your output):

- idf(cash) = ln(6 / 2) + 1 = 2.0986 and idf(today) = ln(6 / 3) + 1 = 1.6931.
- The raw weight is count × idf, so 2.0986 and 1.6931. There is no division by document length.
- The whole D2 row is then scaled to length 1 (L2 norm). Its length is sqrt(4 × 2.0986² + 4 × 1.6931²) = 5.393.
- Final values: 2.0986 / 5.393 = **0.3891** and 1.6931 / 5.393 = **0.3139**.

**Why the numbers differ:** the textbook version I used divides the count by document length, uses plain ln(N/df) with no smoothing and no "+1", and does not normalise the row. scikit-learn uses smoothed idf, adds 1, and normalises the row. The ratio between "cash" and "today" is 0.2012 / 0.1145 = 1.76 by hand but only 0.3891 / 0.314 = 1.24 in scikit-learn, so the "+1" compresses the difference between rare and common words.

## P4. Short answers

**Task 1: One-hot encoding.** "win free cash" becomes three separate vectors of length 6 (a 3×6 block), not one vector for the sentence, so its size depends on sentence length. The vectors for "free" and "cash" share no 1s, so the encoding says nothing about how related words are. Any word outside the 6-word vocabulary (such as "phone") cannot be encoded at all.

**Task 2: Bag of Words.** The matrix has shape (5, 32), and in D1 only "win" has a count of 2. "win free phone" and "phone free win" gave identical vectors (`Vectors equal: True`), which shows that word order is thrown away. Bag of Words keeps what words appear and how often, but not where.

**Task 4: Preprocessing inside the vectorizer.** By default D1 gives 7 features (claim, free, now, phone, reply, to, win), because text is lowercased and "FREE", "WIN" and "Win" merge into "free" and "win". With `lowercase=False` the vocabulary has 8 entries, because "WIN" and "Win" become separate features. With `stop_words='english'`, "now" and "to" disappear, leaving 5 features, and on the full corpus the word "not" is gone (`Is 'not' in vocabulary: False`). Stopword removal therefore shrinks the vocabulary but can delete words that matter, such as the negation in "not good".

**Task 5: Limiting the vocabulary.** `min_df=2` cut the vocabulary from 32 words to 7 (free, lecture, notes, now, the, today, win), and `max_features=5` kept only 5 (free, lecture, now, the, win). Both settings removed words that appear in one document, such as "cash", "prizes" and "claim", which look spam-specific. `max_features=5` kept the uninformative word "the". Several words tie at a total count of 2, so the cut-off among them is somewhat arbitrary.

**Task 6: N-grams.** For D5, unigrams give 9 features, and adding bigrams gives 18, including "not good", "was not" and "good but". On the whole corpus the vocabulary grows from 32 (unigrams) to 67 for (1,2) and 98 for (1,3). The corpus has 41 tokens, which is 36 bigram positions and 31 trigram positions. My arithmetic from your numbers is that 35 of 36 bigrams and all 31 trigrams are distinct, so almost every n-gram occurs only once. That is the sparsity cost of larger n-grams.

**Task 7: Count versus TF-IDF.** All eight words in D2 have count 1, so counts treat them as equally important. TF-IDF gives 0.389 to words that occur only in D2 (call, cash, entry, prizes) and 0.314 to words that also occur in another document (free, now, today, win). This shows how TF-IDF uses information from the whole corpus, not just from the document.

**Task 8: TF-IDF by hand.** My hand values are 0.2012 for "cash" and 0.1145 for "today", while scikit-learn gives 0.3891 and 0.314. The difference comes from three choices: scikit-learn smooths the idf, adds 1 to it, and L2-normalises the row, and the textbook version does none of these. The hand calculation and scikit-learn agree on the main idea, that "cash" (df = 1) is weighted higher than "today" (df = 2).

**Task 10: `fit` and `transform`.** After fitting on the first 4 documents the vocabulary has 26 words. "Claim your free prize now" keeps only claim, free and now, and "prize" and "your" are out-of-vocabulary. "Is the lunch meeting moved to tomorrow?" keeps lunch, meeting, the and to, and loses "is", "moved" and "tomorrow". `transform` drops unseen words silently instead of raising an error, so information is lost without warning. Note that "prize" is lost even though "prizes" is in the vocabulary, because there is no stemming.

**Task 11: Data leakage.** Fitting on training and test texts together added 5 words to the vocabulary (is, moved, prize, tomorrow, your), which are exactly the out-of-vocabulary words from Task 10. In this small count example those extra columns would be all zeros in training, so a linear model would give them no weight, and the harm here is small. The leakage still matters in general: with TF-IDF the idf values depend on test-set document frequencies, and `min_df`, `max_features` and feature selection would also be influenced by the test set. Correct practice is to split first, fit on training data only, and use `transform` on test data.

**Task 12: No meaning.** "tasty" versus "delicious" has cosine similarity 0.75, which is exactly the same as "tasty" versus "terrible" (0.75). A synonym pair and an antonym pair look identical to Bag of Words, because the reviews share three of their four words and the model only sees "same word" or "different word". Counts and TF-IDF cannot represent that "tasty" and "delicious" mean nearly the same thing, which is the main limitation discussed in Part 7 of the report.

## P5. Which settings for a spam detector?

For a spam detector I would use TF-IDF (or counts, tested on a validation set) with `lowercase=True`, no built-in English stopword list, unigrams plus bigrams (`ngram_range=(1, 2)`) with a small `min_df`, and no tight `max_features` cap. Spam relies on specific rare words and short phrases such as "free", "win", "cash", "prizes", "call now" and "free entry": TF-IDF gave the one-document words in D2 (call, cash, entry, prizes) 0.389 against 0.314 for shared words, and my output shows "free", "win" and "now" occurring in spam only. I would avoid `stop_words='english'` because it removed "now" and "to" from D1, and "now" appeared in both spam messages and in none of the three ordinary ones, so it is a useful urgency signal. I would also avoid trigrams (98 features from 5 short documents, all trigrams unique) and tiny vocabulary limits: `max_features=5` kept "the" and dropped "cash", "prizes" and "claim". `min_df=2` cut the vocabulary from 32 to 7 on this tiny corpus, which is too harsh here, but on a large real corpus a small `min_df` (for example 2 to 5) is reasonable for removing typos. Bigrams would add noise on a corpus this small (67 features from 5 messages, with 35 of 36 bigram positions unique), so they only make sense on real data with `min_df`. `lowercase=False` split "WIN" and "Win" into separate features; I would keep lowercasing for the vocabulary and, if capital letters matter for spam, add a separate feature such as the fraction of capitalised words. Finally, "prize" in a new message was out-of-vocabulary even though "prizes" was in the vocabulary, so stemming or character n-grams are worth testing (this is my suggestion, not something my script ran), and the vectorizer must be fitted on training data only, as shown in Task 11.
