# Assignment 3 – Vectorization

**Student:** Abdrihman Hussein

## Part A – Research

### 1. Text Vectorization and One-Hot Encoding

**Text vectorization** is the process of turning a document into a numerical vector so that a machine-learning model can process it. A vectorizer defines a vocabulary $V = \{t_1, t_2, \ldots, t_{|V|}\}$ and a mapping $\phi: d \mapsto \mathbb{R}^{|V|}$ that represents every document with the same number of features. The vocabulary and the mapping are learned from data, not chosen by hand. This is why vectorization is a form of **feature extraction**: the raw text is replaced by a fixed set of measurements that the model can weight, compare and optimise (Salton & McGill, 1983; Manning et al., 2008).

Most machine-learning models need **the same number of numerical features for every example** for three reasons. First, a linear model, a neural network or a Naive Bayes classifier stores one parameter per feature. If document A has 7 features and document B has 12, there is no consistent parameter vector to learn. Second, training algorithms operate on matrices: rows are examples and columns are features, and matrix multiplication requires a rectangular shape. Third, a column must mean the same thing in every row. If column 4 means `free` in one document and `phone` in another, the learned weight for column 4 has no stable interpretation. Fixed dimensionality is therefore not a cosmetic choice; it is what makes shared parameters, distances and regularisation meaningful (Jurafsky & Martin, 2025).

**One-hot encoding of words** is the simplest vector representation. For a vocabulary of size $|V|$, each word $w$ is represented by a vector with a 1 in the position of $w$ and 0 everywhere else:

$$
\text{one-hot}(w)_j =
\begin{cases}
1 & \text{if } t_j = w,\\
0 & \text{otherwise.}
\end{cases}
$$

Using the vocabulary from Task 1, `[cash, class, free, lunch, notes, win]`, the word `free` becomes `[0, 0, 1, 0, 0, 0]` and `cash` becomes `[1, 0, 0, 0, 0, 0]`. Every word is equidistant from every other word: the dot product of two different one-hot vectors is 0, and the Euclidean distance is always $\sqrt{2}$. The representation says *which* word is present, but nothing about how similar two words are.

One-hot encoding is **not enough to represent a whole document**. A document normally contains several words, so encoding each word separately produces a *sequence* of vectors whose length equals the number of words. That sequence has a different shape for every document, and a classifier expects one fixed-length vector per document. Concatenating the word vectors makes the dimension depend on document length; padding shorter documents creates enormous sparse vectors and position-specific weights that do not transfer to new sentences. Summing or averaging the vectors produces a document vector, but that is already the Bag-of-Words idea in the next section, and it discards order. One-hot vectors also cannot express frequency (`win` once and `win` three times look the same), cannot express similarity (`car` and `bus` are as different as `car` and `cake`), and become impractical when the vocabulary contains 100,000 words (Harris, 1954; Manning et al., 2008).

A real-life example makes the shape problem concrete. A phone spam filter must classify every incoming SMS, and messages have arbitrary length. The message `Win` produces one one-hot vector, while `Win a FREE phone now! Reply WIN to claim.` produces nine word tokens. The filter cannot accept a different-sized input for every message. It needs one vector per message, which is why practical systems aggregate word information into a document-term vector or a dense embedding.

| Representation | Shape for one document | What it captures | Main limitation |
|---|---|---|---|
| One-hot word | $1 \times |V|$ | Identity of one word | No frequency, no similarity |
| Concatenated one-hot document | $1 \times (|V| \times L)$ | Word identity and position | Length-dependent, extremely sparse |
| Bag of Words / document-term matrix | $1 \times |V|$ | Counts of words | Word order and meaning lost |
| TF-IDF | $1 \times |V|$ | Weighted importance of words | Word order and meaning lost |
| Dense embedding | $1 \times d$ | Learned semantic similarity | Needs a model and training data |

### 2. Bag of Words and the Document-Term Matrix

**Bag of Words (BoW)** represents a document as the multiset of its tokens. The model records which words occur and how many times, but throws away grammar, syntax and word order. The name is literal: if the words of a sentence are placed in a bag, one can count them but cannot reconstruct the sentence. The **document-term matrix (DTM)** is the standard storage format for BoW. It has one row per document, one column per vocabulary term, and one cell per document-term pair. Cell $(i, j)$ is the number of times term $j$ occurs in document $i$ (or a weight such as TF-IDF). The rows therefore describe documents, the columns describe the vocabulary, and the cells describe presence, frequency or importance (Manning et al., 2008; Sebastiani, 2002).

A small example shows the structure:

| Document | free | win | phone | lunch | notes |
|---|---:|---:|---:|---:|---:|
| D1: Win a FREE phone now! Reply WIN to claim. | 1 | 2 | 1 | 0 | 0 |
| D2: Free entry: win cash prizes today. Call now. | 1 | 1 | 0 | 0 | 0 |
| D3: Are we still meeting for lunch today? | 0 | 0 | 0 | 1 | 0 |
| D4: Please send me the lecture notes before class. | 0 | 0 | 0 | 0 | 1 |

The matrix is called a **bag** because the order of the words is not represented. `win free phone` and `phone free win` produce exactly the same row, even though one is a plausible spam phrase and the other is not. In the practical task, the two phrases produced identical vectors, which is the clearest demonstration of the loss.

**Where losing order does little harm.** Topic classification and spam detection often depend on the presence of a few discriminative content words. A news article about football contains `goal`, `match`, `team` and `coach`; a spam message contains `free`, `prize`, `claim` and `win`. The exact sentence order is less important than the vocabulary. For long documents, the counts of content words are a strong signal, and BoW is still a competitive baseline for topic categorisation and information retrieval (Joachims, 1998; Sebastiani, 2002).

**Where losing order causes real problems.** Sentiment analysis and negation are the standard counter-example. `The food was good` and `The food was not good` contain almost the same bag of words; the only difference is `not`, and a model that treats `not` as just another word can read the second sentence as positive. The problem is even clearer with `dog bites man` and `man bites dog`: both have the same BoW vector, but one is ordinary news and the other is unusual enough to be a headline. `New York` and `York New` have the same bag but different meanings. In a spam filter, `not free` and `free` may both contain `free`, so a purely lexical model can be fooled by a negated claim.

| Task | Does order matter? | Why |
|---|---|---|
| Topic classification of long news articles | Little | Content-word counts are usually enough to identify the topic |
| Spam detection with obvious trigger words | Little | `free`, `prize`, `claim`, `win` are strong cues wherever they appear |
| Sentiment analysis with negation | A lot | `not good` is not the same as `good` |
| Question answering and intent detection | A lot | `how old are you` and `are you old` ask different things |
| Relation extraction | A lot | `A caused B` and `B caused A` are different relations |

### 3. Vocabulary Size and Sparsity

The **vocabulary size** $|V|$ is the number of distinct terms that the vectorizer keeps after tokenisation and preprocessing. It grows with the size and diversity of the corpus, the domain, the language's morphology, spelling variation, and the use of n-grams. A five-document toy corpus can have a vocabulary of 32 words, as in this assignment; a real collection of news or web pages can easily have 100,000 to several million distinct terms. The document-term matrix therefore has shape $N \times |V|$, where $N$ is the number of documents.

Real document-term matrices are **large and mostly zero** because each document uses only a small fraction of the vocabulary. If a corpus has 1,000,000 documents and a vocabulary of 100,000 terms, the dense matrix has $10^{11}$ cells. At four bytes per cell that is about 400 GB, even though most cells are zero. If each document contains about 100 unique terms, the matrix has only $10^8$ non-zero cells, so roughly 99.9% of the cells are zero. Storing the zeros is wasteful, so sparse matrices store only the non-zero values and their coordinates.

**Sparse storage.** The most common format is **Compressed Sparse Row (CSR)**. It uses three arrays: `data` stores the non-zero values, `indices` stores the column index of each value, and `indptr` stores the start and end position of each row in the other two arrays. Row $i$ occupies the slice from `indptr[i]` to `indptr[i+1]`. **Compressed Sparse Column (CSC)** does the same by columns, which is convenient for column-wise operations such as computing document frequencies. **Coordinate list (COO)** stores triples `(row, column, value)` and is easy to build but less efficient for arithmetic. The memory cost is $O(\text{nnz})$ instead of $O(N \times |V|)$, where `nnz` is the number of non-zero entries. scikit-learn's vectorizers return CSR matrices by default, and `matrix.nnz` reports the number of stored non-zeros (Pedregosa et al., 2011; Aggarwal & Zhai, 2012).

In the practical task, the count matrix has $5 \times 32 = 160$ cells, 39 non-zero cells and 121 zero cells, so 75.62% of the matrix is zero. That percentage is modest because the corpus is tiny; in a real corpus the zero share is usually above 99%.

**Settings that change the vocabulary.** The vectorizer's parameters and preprocessing choices decide which terms survive:

| Setting or choice | Effect on the vocabulary | Benefit | Risk |
|---|---|---|---|
| `min_df` | Drops terms that appear in fewer than a minimum number or fraction of documents | Removes typos, hapax legomena and noise; reduces overfitting | Can drop rare but informative terms such as `cash` or `prizes` |
| `max_df` | Drops terms that appear in more than a maximum number or fraction of documents | Removes corpus-wide words such as `the`, `email` or `dear` | Can drop frequent domain words that still carry signal |
| `max_features` | Keeps only the $k$ most frequent terms | Hard cap on memory and training time | Can drop rare discriminative words; ties are broken arbitrarily |
| Lowercasing | Merges `Win`, `WIN` and `win` into one term | Smaller vocabulary; better matching | Loses case cues such as `URGENT` or `Apple` versus `apple` |
| Stopword removal | Removes a fixed list of frequent function words | Smaller, less noisy matrix | Can remove negation (`not`), time words (`now`) or content in short texts |
| Stemming | Strips suffixes so `connect`, `connected` and `connecting` share a stem | Smaller vocabulary; more counts per feature | Can conflate different words (`university` and `universal` both become `univers`) |
| Lemmatisation | Maps words to dictionary lemmas (`was` to `be`, `better` to `good`) | Linguistically accurate reduction | Slower; needs a dictionary or part-of-speech information |
| N-grams | Adds sequences of two or more tokens | Keeps local order and phrases | Vocabulary grows quickly; most n-grams are rare |

**Stemming and lemmatisation** are the two standard ways to reduce morphological variation. Porter's stemmer applies a fixed set of suffix-stripping rules and is fast and language-independent in style, but it can produce non-words and can merge unrelated terms (Porter, 1980; Lovins, 1968; Bird et al., 2009). Lemmatisation uses a dictionary and usually part-of-speech information to return a valid lemma, so it is more accurate but slower. Both reduce $|V|$ and increase the count per remaining feature, which helps when the corpus is small.

**Stopword removal** is particularly risky in short texts. Fox (1990) and Wilbur and Sirotkin (1992) describe stop lists as a way to remove words with little topical content, but the definition of "little content" depends on the task. In this assignment, scikit-learn's English stop list removed `now` and `to` from D1, and it removed `not` from D5. The last removal changes `The lecture was not good, but the notes were great` into `lecture good notes great`, which reads like a positive review. A spam filter should therefore be careful with stopword removal, because `not`, `now`, `to` and similar words can be part of the signal.

**Feature selection** is the broader version of these limits. Yang and Pedersen (1997) and Forman (2003) show that removing low-information terms can improve both speed and accuracy, but the best criterion depends on the classifier and the class balance. A vocabulary limit is not automatically good; it is a trade between noise reduction and information loss.

### 4. N-grams

An **n-gram** is a contiguous sequence of $n$ tokens. A **unigram** is one token (`good`), a **bigram** is two adjacent tokens (`not good`), and a **trigram** is three (`was not good`). A vectorizer with `ngram_range=(1, 2)` keeps both unigrams and bigrams, so the feature set contains single words and two-word phrases. N-grams keep part of the word order because the pair or triple is treated as one feature; `not good` is no longer the same as `good` plus `not` (Jurafsky & Martin, 2025; Manning et al., 2008).

A concrete example is D5: `The lecture was not good, but the notes were great.` The unigram vocabulary contains `good`, `not`, `lecture`, `notes`, `great` and the other single words. The unigram-plus-bigram vocabulary adds `the lecture`, `lecture was`, `was not`, `not good`, `good but`, `but the`, `the notes`, `notes were` and `were great`. The bigram `not good` keeps the reviewer's negative opinion; without it, a model that sees `good` may treat the review as positive. The same logic applies to `New York` versus `York New`, and to `dog bites man` versus `man bites dog`: the bag of unigrams is identical, but the bigrams are different.

The cost of larger n-grams is **combinatorial growth and sparsity**. In the worst case, the number of possible n-grams up to order $n$ is $\sum_{k=1}^{n} |V|^k$, which is dominated by $|V|^n$. For a vocabulary of 32 words, the upper bound for trigrams is $32 + 32^2 + 32^3 = 33{,}824$ features, even though the actual count on this tiny corpus is only 98. In a real corpus with $|V| = 50{,}000$, trigrams would be impossible to store densely. Most n-grams also occur only once or twice, so their counts are unreliable and the model needs much more data to estimate them. Longer n-grams capture longer phrases but are rarer; this is why language models use smoothing and backoff, and why practical text classifiers usually stop at bigrams or trigrams (Jurafsky & Martin, 2025).

| n | Example from D5 | What it adds | Main cost |
|---|---|---|---|
| 1 (unigram) | `good`, `not`, `notes` | Individual word identity | No order, no negation |
| 2 (bigram) | `not good`, `the notes`, `were great` | Local word order and phrases | Vocabulary grows roughly quadratically; many rare features |
| 3 (trigram) | `was not good`, `the notes were` | Longer context and idioms | Vocabulary grows roughly cubically; most trigrams are very rare |

In the practical output, the vocabulary size on D1–D5 grows from 32 for unigrams to 67 for unigrams plus bigrams and 98 for unigrams, bigrams and trigrams. The growth is not exactly multiplicative because the corpus is tiny and many combinations never occur, but the direction is clear: each increase in $n$ adds many features and makes the matrix sparser.

### 5. TF-IDF

**TF-IDF** stands for **term frequency–inverse document frequency**. It is a weighting scheme that gives a word a high weight in a document when the word occurs often in that document but rarely in the rest of the corpus. The intuition is that a word which appears in every document (such as `the`) cannot distinguish documents, while a word that appears in only a few documents is more specific and therefore more useful (Salton & McGill, 1983; Manning et al., 2008).

The idea of **inverse document frequency** comes from information retrieval. Sparck Jones (1972) argued that term specificity should be measured by the number of documents in which a term occurs, and proposed the inverse document frequency as a statistical interpretation of that specificity. Robertson and Sparck Jones (1976) developed the probabilistic relevance weighting framework, and Salton and McGill (1983) popularised the TF-IDF combination in vector-space retrieval. Luhn (1958) had earlier used term frequency as a measure of term importance, so TF-IDF can be seen as the combination of a local frequency signal and a global rarity signal.

The **textbook formula** is:

$$
\text{tf}(t,d) = \frac{\text{count}(t,d)}{|d|},
\qquad
\text{idf}(t) = \log\frac{N}{\text{df}(t)},
\qquad
\text{tf-idf}(t,d) = \text{tf}(t,d) \times \text{idf}(t),
$$

where $N$ is the number of documents and $\text{df}(t)$ is the number of documents that contain term $t$. Some textbooks use the raw count instead of the length-normalised frequency; the important part is that the weight combines local frequency with global rarity. If a term appears in every document, $\text{df}(t) = N$ and $\text{idf}(t) = \log 1 = 0$, so the term is effectively ignored. If a term appears in only one document, $\text{idf}(t) = \log N$, which is the largest possible value. **Rarer words therefore get higher weights**, provided they occur in the document.

**How scikit-learn differs from the textbook formula.** `TfidfVectorizer` uses three adjustments by default (Pedregosa et al., 2011; scikit-learn developers, 2024):

1. **Smoothing.** The IDF is
   $$
   \text{idf}(t) = \log\frac{1 + N}{1 + \text{df}(t)} + 1,
   $$
   not $\log(N/\text{df}(t))$. The `+1` in the numerator and denominator avoids division by zero and avoids a zero IDF for a term that appears in every document. The outer `+1` keeps all IDF values positive, so a term that appears everywhere still has a small non-zero weight.
2. **Term frequency.** The default is the raw count, not the length-normalised frequency. Setting `sublinear_tf=True` replaces the count with $1 + \log(\text{count})$, which dampens the effect of very frequent words.
3. **Normalisation.** Each document vector is scaled to unit L2 norm by default (`norm="l2"`). This removes the effect of document length: a long document and a short document that use the same words in the same proportions get the same direction. It also makes the dot product of two rows equal to their cosine similarity.

Because of these adjustments, scikit-learn's numbers differ from a hand calculation with $\log(N/\text{df})$. In the practical task, D2 gives a textbook weight of 0.2012 for `cash` and 0.1145 for `today`, while scikit-learn gives 0.3891 and 0.3140. The ranking is the same because `cash` occurs in one document and `today` occurs in two, so `cash` has the higher IDF in both formulas.

**TF-IDF values are weights, not probabilities.** A weight of 0.57 does not mean a 57% chance that the document belongs to a class. TF-IDF has no sample space, no event to assign a probability to, and no requirement that the values sum to one. The weights are relative importance scores: they say how strongly a term represents a document compared with the rest of the corpus. They depend on the corpus, the preprocessing, the smoothing and the normalisation, so they are not comparable across datasets. To obtain probabilities, a classifier such as Naive Bayes or logistic regression must be trained on the TF-IDF features; the TF-IDF values themselves are only the input features (Manning et al., 2008; scikit-learn developers, 2024).

| Component | Textbook TF-IDF | scikit-learn default | Effect |
|---|---|---|---|
| Term frequency | Count divided by document length, or raw count | Raw count (`sublinear_tf=False`) | Long documents are not normalised until the final step |
| Inverse document frequency | $\log(N / \text{df})$ | $\log((1+N)/(1+\text{df})) + 1$ | Smoothing avoids zero and negative IDF; all weights stay positive |
| Normalisation | None | L2 row normalisation | Document length is removed; cosine similarity becomes a dot product |
| Output | Weight | Weight | Not a probability and not comparable across corpora |

### 6. `fit`, `transform`, `fit_transform`, Out-of-Vocabulary Words and Data Leakage

The three methods describe three different operations:

- **`fit`** learns from the training text. For a count vectorizer it builds the vocabulary and the document frequencies; for a TF-IDF vectorizer it also computes the IDF values. It returns the fitted vectorizer and does not produce a document-term matrix.
- **`transform`** applies an already fitted vectorizer to text. It maps each document to a row using the learned vocabulary and IDF values. Words that are not in the vocabulary are ignored, because there is no column for them.
- **`fit_transform`** performs `fit` and `transform` on the same data in one call. It is efficient for the training set, but it must not be used on the test set, because that would learn the test vocabulary and test document frequencies.

A useful analogy is a dictionary. `fit` is the process of building the dictionary from a set of documents; `transform` is looking up words in an existing dictionary; `fit_transform` builds the dictionary and immediately uses it on the same documents. The dictionary must be built from training data only, otherwise the test data has influenced the feature space.

**Out-of-vocabulary (OOV) words.** When `transform` sees a token that was not in the training vocabulary, it cannot assign it a column, so the token contributes nothing. In the practical task, a vectorizer fitted on D1–D4 does not know `prize`, because D2 contains `prizes` and the vectorizer has no stemming or lemmatisation. It also does not know `your`, `is`, `moved` or `tomorrow`. The transformed vectors for N1 and N2 therefore contain zeros for those words. This is a serious limitation for short messages and for morphologically rich languages, where a new word form can carry the whole meaning of a message (Manning et al., 2008; Jurafsky & Martin, 2025).

There are several ways to reduce OOV loss. **Stemming or lemmatisation** maps `prizes` and `prize` to the same stem or lemma, so the word is no longer unknown. **Character n-grams** represent a word by its character sequences, so `prize` and `prizes` share most of their features. **Subword tokenisation** such as Byte-Pair Encoding or SentencePiece splits rare words into reusable pieces and gives every string a representation (Sennrich et al., 2016; Kudo & Richardson, 2018). **HashingVectorizer** maps any token to a fixed number of buckets without storing a vocabulary, so there is no true OOV, but collisions can mix unrelated words. **Dense embeddings** represent unseen words through their subword pieces or their context (Mikolov et al., 2013).

**Data leakage in vectorization.** Leakage occurs when information that would not be available at prediction time influences training or evaluation (Kaufman et al., 2012). In vectorization, the classic mistake is to fit the vectorizer on the whole dataset, including the test set, and then split the resulting matrix. The vocabulary then contains test words, and the IDF values are computed from test document frequencies. The model has indirectly seen the test set. The test score looks better than it really is, because the feature space was built with knowledge of the test data. The problem is worst with small datasets, duplicate or near-duplicate documents, and rare terms that appear only in the test set: those terms get their own columns and can act as a shortcut.

The **correct order** is:

1. Split the data into training and test sets first.
2. Fit the vectorizer on `X_train` only.
3. Transform `X_train` and `X_test` with that fitted vectorizer.
4. Train the classifier on the transformed training data.
5. Evaluate on the transformed test data.
6. For cross-validation, fit the vectorizer inside each fold, not once on the full training set.

A scikit-learn `Pipeline` enforces this order automatically: `Pipeline([("vectorizer", TfidfVectorizer()), ("model", LogisticRegression())])` fits the vectorizer only on the training fold and transforms the validation or test fold with the learned vocabulary. This is the safest way to prevent leakage in a real project (Pedregosa et al., 2011).

In the practical task, the vectorizer fitted on D1–D4 has 26 words. The vectorizer fitted on D1–D4 plus N1 and N2 has 31 words. The five extra words are `is`, `moved`, `prize`, `tomorrow` and `your`, all of which come from the test texts. If those columns were available during training, the model could learn weights for words that only exist in the test set, and the evaluation would no longer measure generalisation to unseen text.

| Step | Data used | What is learned | Why it matters |
|---|---|---|---|
| `fit` on training text | `X_train` only | Vocabulary, document frequencies, IDF | The feature space is defined without test knowledge |
| `transform` training text | `X_train` | Training matrix | The classifier learns from the training features |
| `transform` test text | `X_test` | Test matrix | Unseen words are ignored; the test set stays independent |
| `fit_transform` on all text | `X_train` + `X_test` | Vocabulary and IDF from both | Leakage: the test set influences the features and the score becomes optimistic |

### 7. Case Study: Somali News Classification with TF-IDF and a Linear SVM

The case study is Mohamed Mohamud Ali's study of **Somali news classification**, published in *Scientific Reports* in 2026 and available earlier as a full-text preprint (Ali, 2026a; Ali, 2026b). The study is directly relevant because it deals with a low-resource African language and compares classical sparse text representations with unsupervised and supervised alternatives on a real classification task.

**Task and data.** The task is to assign a Somali news story to one of eight thematic categories: `Caafimaad` (health), `Ciyaaro` (sports), `Dagaal` (war), `Diini` (religion), `Ganacsi` (business), `Madadaalo` (entertainment), `Siyaasad` (politics) and `Tiknoolajiyad` (technology). The author built a manually curated benchmark corpus of **7,240 Somali news stories**, scraped from BBC Somali and VOA Somali and annotated by native-speaker domain experts. Somali is a low-resource, agglutinative language without a standardised digital orthography, so the corpus is both a linguistic and a computational challenge (Ali, 2026a).

**Representations compared.** The main pipeline represents each document with **word-level TF-IDF unigram features**, using tailored normalisation and a custom Somali stopword list that removes low-information particles such as `waa`, `iyo` and `ku`, and classifies with a **linear Support Vector Machine** in a one-vs-rest scheme for the eight classes (Ali, 2026a; Ali, 2026b). The published study benchmarks this pipeline against **Multinomial Naive Bayes**, **Logistic Regression**, **Random Forest** and **alternative SVM n-gram variants** on the same features (Ali, 2026a). The preprint additionally reports two unsupervised baselines: **Latent Dirichlet Allocation (LDA)**, which models each document as a mixture of latent topics (Blei et al., 2003), and **Latent Semantic Indexing (LSI)**, which projects the term-document matrix into a lower-dimensional space with a truncated singular-value decomposition (Deerwester et al., 1990) (Ali, 2026b).

**Results.** In a stratified 80/20 holdout evaluation the best linear SVM configuration reached **94.20% accuracy**, and repeated stratified holdout and stratified five-fold cross-validation gave mean accuracies of **93.11%** and **93.20%**, so the performance is stable across data partitions rather than the product of one lucky split; the linear SVM also outperformed Naive Bayes, Logistic Regression, Random Forest and the n-gram SVM variants under identical preprocessing (Ali, 2026a). In the preprint both unsupervised baselines were rejected: LDA produced overlapping, low-coherence topics and LSI showed deficient cluster separation, so neither recovered the eight editorial categories (Ali, 2026b). In that version the per-category F1 scores were highest for `Diini` (0.98) and `Tiknoolajiyad` (0.97), where the vocabulary is highly specialised, and lowest for `Siyaasad` (0.90), because political stories share many tokens with `Dagaal` (Ali, 2026b). The author argues that such a lightweight TF-IDF plus SVM pipeline is a resource-efficient alternative to deep-learning systems in low-resource settings (Ali, 2026a).

| Representation | What it captures | Result in the study |
|---|---|---|
| TF-IDF + linear SVM | Sparse lexical features weighted by rarity; no word order | 94.20% holdout accuracy; 93.11% and 93.20% mean accuracy on repeated holdout and five-fold cross-validation |
| Multinomial Naive Bayes, Logistic Regression, Random Forest | The same TF-IDF features with different decision rules | Outperformed by the linear SVM |
| SVM with n-gram variants | TF-IDF features plus local word order | Outperformed by the unigram linear SVM |
| LDA topic vectors | Latent topic proportions; no word order | Rejected: overlapping, low-coherence topics (preprint) |
| LSI latent vectors | Lower-dimensional co-occurrence structure | Rejected: deficient cluster separation (preprint) |

**How the study shows the main limitation of BoW and TF-IDF.** The TF-IDF pipeline treats every word form as an independent column. It does not know that two different words can mean the same thing, that one word can have several meanings, or that a negated phrase means the opposite of its positive form. Somali's complex morphology and orthographic variation make this limitation especially visible: the pipeline relies on a whitespace tokeniser and a custom stopword list without stemming or lemmatisation (Ali, 2026b), so different surface forms of the same root, or the same word written with different regional spellings, become different features, and a model trained on one form has no direct way to recognise the other. The clearest evidence of the lexical limitation is the confusion between `Siyaasad` and `Dagaal`: political and conflict stories share vocabulary, and because a bag of words cannot represent the relation between actors and events, the classifier's margins overlap exactly there (Ali, 2026b). The LDA and LSI representations try to capture co-occurrence and latent structure, but they performed poorly on this corpus, so they did not solve the problem either. Dense word embeddings would represent similar words with similar vectors (Mikolov et al., 2013), and subword representations would reduce the out-of-vocabulary problem (Sennrich et al., 2016), but they require more data and computation. The 94.20% accuracy is impressive, yet it measures lexical overlap on a corpus whose test split shares most of its vocabulary with the training split; it does not prove that the model understands Somali, only that vocabulary is a strong signal for this task.

The practical task in this assignment shows the same limitation on a tiny scale. The reviews `the food was tasty`, `the food was delicious` and `the food was terrible` share the words `the`, `food` and `was`, so all three pairwise cosine similarities are 0.75. `tasty` and `delicious` are synonyms, but the count vectorizer sees them as unrelated columns; `terrible` is the opposite of `tasty`, but it is equally unrelated. The representation captures word identity, not meaning. This is exactly the gap that the Somali study discusses, and it is why Bag of Words and TF-IDF remain useful baselines rather than complete solutions.

## Part B – Practical

### Setup

The script is `vectorization_practice.py`. It uses scikit-learn and NumPy only, and it prints one heading per task. The full output was saved with:

```text
python vectorization_practice.py > output.txt
```

Every number below is copied from `output.txt`; none was typed by hand.

### Task 1 – One-hot encoding

```text
=== 1) ONE-HOT ENCODING ===
Vocabulary: ['cash', 'class', 'free', 'lunch', 'notes', 'win']

Word by word for 'free cash':
  free   -> [0, 0, 1, 0, 0, 0]
  cash   -> [1, 0, 0, 0, 0, 0]

Sentence 'win free cash':
  win    -> [0, 0, 0, 0, 0, 1]
  free   -> [0, 0, 1, 0, 0, 0]
  cash   -> [1, 0, 0, 0, 0, 0]
Number of vectors: 3 (one per word)
```

**Answer.** `free` is `[0, 0, 1, 0, 0, 0]` because it is the third word in the vocabulary, and `cash` is `[1, 0, 0, 0, 0, 0]` because it is the first. The sentence `win free cash` produces three vectors, one per word, so the number of vectors equals the sentence length. A whole document would therefore produce a variable-length list of vectors, not the single fixed-length vector that a classifier needs.

### Task 2 – Bag of Words

```text
=== 2) BAG OF WORDS AND THE DOCUMENT-TERM MATRIX ===
Vocabulary: ['are', 'before', 'but', 'call', 'cash', 'claim', 'class', 'entry', 'for', 'free', 'good', 'great', 'lecture', 'lunch', 'me', 'meeting', 'not', 'notes', 'now', 'phone', 'please', 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today', 'was', 'we', 'were', 'win']
Shape (documents, words): (5, 32)

Document-term matrix (rows = documents, columns = vocabulary):
         are before but call cash claim class entry for free good great lecture lunch me meeting not notes now phone please prizes reply send still the to today was we were win
D1         0      0   0    0    0     1     0     0   0    1    0     0       0     0  0       0   0     0   1     1      0      0     1    0     0   0  1     0   0  0    0   2
D2         0      0   0    1    1     0     0     1   0    1    0     0       0     0  0       0   0     0   1     0      0      1     0    0     0   0  0     1   0  0    0   1
D3         1      0   0    0    0     0     0     0   1    0    0     0       0     1  0       1   0     0   0     0      0      0     0    0     1   0  0     1   0  1    0   0
D4         0      1   0    0    0     0     1     0   0    0    0     0       1     0  1       0   0     1   0     0      1      0     0    1     0   1  0     0   0  0    0   0
D5         0      0   1    0    0     0     0     0   0    0    1     1       1     0  0       0   1     1   0     0      0      0     0    0     0   2  0     0   1  0    1   0
```

**Document-term matrix as a table:**

| Document | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| D2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| D3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| D4 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| D5 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |

**Answer.** The cell in D1 with the value `2` is the `win` column, because `WIN` appears twice in `Win a FREE phone now! Reply WIN to claim.` The phrases `win free phone` and `phone free win` produce identical vectors (`win` = 1, `free` = 1, `phone` = 1, everything else 0), because Bag of Words counts words and discards their order. The two phrases are different sentences, but the model cannot see the difference.

### Task 3 – Tokens vs vocabulary size

```text
=== 3) TOKENS vs VOCABULARY SIZE ===
Tokens per document (vectorizer analyser): [8, 8, 7, 8, 10]
Total tokens (vectorizer analyser): 41
Total tokens (plain whitespace split): 42
Vocabulary size: 32
Words that appear in more than one document: ['free', 'lecture', 'notes', 'now', 'the', 'today', 'win']
```

**Answer.** The vectorizer sees 41 tokens across the five documents, but only 32 distinct words. The difference between 41 and 42 is caused by the tokenizer: the plain whitespace split counts the one-letter word `a` and the punctuation attached to words, while the vectorizer's default pattern keeps only tokens of at least two word characters. The seven words that appear in more than one document are `free`, `lecture`, `notes`, `now`, `the`, `today` and `win`; these are the terms that `min_df=2` will keep.

### Task 4 – Preprocessing inside the vectorizer

```text
=== 4) PREPROCESSING INSIDE THE VECTORIZER (D1 ONLY) ===
D1: Win a FREE phone now! Reply WIN to claim.
Default:               ['claim', 'free', 'now', 'phone', 'reply', 'to', 'win']
lowercase=False:       ['FREE', 'Reply', 'WIN', 'Win', 'claim', 'now', 'phone', 'to']
stop_words='english':  ['claim', 'free', 'phone', 'reply', 'win']

What happened to 'a': removed
What happened to '!': removed
Default forms of win: ['win']
lowercase=False forms of win: ['Win', 'WIN']
Stopwords removed from D1: ['now', 'to']

stop_words='english' on D1-D5 vocabulary: ['cash', 'claim', 'class', 'entry', 'free', 'good', 'great', 'lecture', 'lunch', 'meeting', 'notes', 'phone', 'prizes', 'reply', 'send', 'today', 'win']
'not' removed from the vocabulary: True
D5 with stopwords removed: ['lecture', 'good', 'notes', 'great']
```

**Answer.** The default vectorizer lowercases the text, so `Win` and `WIN` both become `win`, and it drops `a` and `!` because its token pattern requires at least two word characters. With `lowercase=False`, `Win` and `WIN` remain two separate features, and `FREE` keeps its capitals. The stopword list removes `now` and `to` from D1; `a` was already removed by the token pattern, so the stopword list has no visible effect on it. On D1–D5, the stopword list removes `not`, which turns D5 into `lecture good notes great`; the negation is gone, so a sentiment model could read the review as positive. This is a clear case where stopword removal destroys meaning.

### Task 5 – Limiting the vocabulary

```text
=== 5) LIMITING THE VOCABULARY ===
Full vocabulary size: 32
Full vocabulary:       ['are', 'before', 'but', 'call', 'cash', 'claim', 'class', 'entry', 'for', 'free', 'good', 'great', 'lecture', 'lunch', 'me', 'meeting', 'not', 'notes', 'now', 'phone', 'please', 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today', 'was', 'we', 'were', 'win']
min_df=2 size: 7
min_df=2:      ['free', 'lecture', 'notes', 'now', 'the', 'today', 'win']
max_features=5 size: 5
max_features=5:      ['free', 'lecture', 'now', 'the', 'win']
```

**Answer.** `min_df=2` keeps only words that appear in at least two documents, so the vocabulary falls from 32 to 7 and contains the shared words `free`, `lecture`, `notes`, `now`, `the`, `today` and `win`. `max_features=5` keeps the five most frequent terms, which are `free`, `lecture`, `now`, `the` and `win`. Both limits remove rare words: `min_df=2` drops one-document words such as `cash`, `prizes`, `claim` and `phone`, and `max_features=5` drops most of the vocabulary, including informative words such as `cash` and `prizes`, while keeping the function word `the`. The surviving words are therefore the frequent or shared ones, not necessarily the most useful ones for a classifier.

### Task 6 – N-grams

```text
=== 6) N-GRAMS ===
D5: The lecture was not good, but the notes were great.
Unigrams (1,1):         ['but', 'good', 'great', 'lecture', 'not', 'notes', 'the', 'was', 'were']
Unigrams + bigrams (1,2): ['but', 'but the', 'good', 'good but', 'great', 'lecture', 'lecture was', 'not', 'not good', 'notes', 'notes were', 'the', 'the lecture', 'the notes', 'was', 'was not', 'were', 'were great']

Vocabulary size on D1-D5:
  ngram_range=(1, 1): 32
  ngram_range=(1, 2): 67
  ngram_range=(1, 3): 98
```

**Answer.** The bigram `not good` keeps the reviewer's negative opinion, because it ties the negation to the adjective; without it, the unigram `good` could dominate. The vocabulary grows from 32 unigrams to 67 unigrams plus bigrams and 98 unigrams plus bigrams plus trigrams on this small corpus. In general the number of possible n-grams grows roughly like $|V|^n$, and most of the added features occur only once, so larger n-grams cost memory, training time and statistical reliability.

### Task 7 – TF-IDF

```text
=== 7) TF-IDF ===
TF-IDF matrix (rounded to 2 decimals):
          are before  but call cash claim class entry  for free good great lecture lunch   me meeting  not notes  now phone please prizes reply send still  the   to today  was   we were  win
D1       0.00   0.00 0.00 0.00 0.00  0.36  0.00  0.00 0.00 0.29 0.00  0.00    0.00  0.00 0.00    0.00 0.00  0.00 0.29  0.36   0.00   0.00  0.36 0.00  0.00 0.00 0.36  0.00 0.00 0.00 0.00 0.57
D2       0.00   0.00 0.00 0.39 0.39  0.00  0.00  0.39 0.00 0.31 0.00  0.00    0.00  0.00 0.00    0.00 0.00  0.00 0.31  0.00   0.00   0.39  0.00 0.00  0.00 0.00 0.00  0.31 0.00 0.00 0.00 0.31
D3       0.39   0.00 0.00 0.00 0.00  0.00  0.00  0.00 0.39 0.00 0.00  0.00    0.00  0.39 0.00    0.39 0.00  0.00 0.00  0.00   0.00   0.00  0.00 0.00  0.39 0.00 0.00  0.31 0.00 0.39 0.00 0.00
D4       0.00   0.38 0.00 0.00 0.00  0.00  0.38  0.00 0.00 0.00 0.00  0.00    0.31  0.00 0.38    0.00 0.00  0.31 0.00  0.00   0.38   0.00  0.00 0.38  0.00 0.31 0.00  0.00 0.00 0.00 0.00 0.00
D5       0.00   0.00 0.32 0.00 0.00  0.00  0.00  0.00 0.00 0.00 0.32  0.32    0.26  0.00 0.00    0.00 0.32  0.26 0.00  0.00   0.00   0.00  0.00 0.00  0.00 0.51 0.00  0.00 0.32 0.00 0.32 0.00

D1 words with count and TF-IDF weight:
  claim      count: 1 | tf-idf: 0.356
  free       count: 1 | tf-idf: 0.287
  now        count: 1 | tf-idf: 0.287
  phone      count: 1 | tf-idf: 0.356
  reply      count: 1 | tf-idf: 0.356
  to         count: 1 | tf-idf: 0.356
  win        count: 2 | tf-idf: 0.574

D2 words with count and TF-IDF weight:
  call       count: 1 | tf-idf: 0.389
  cash       count: 1 | tf-idf: 0.389
  entry      count: 1 | tf-idf: 0.389
  free       count: 1 | tf-idf: 0.314
  now        count: 1 | tf-idf: 0.314
  prizes     count: 1 | tf-idf: 0.389
  today      count: 1 | tf-idf: 0.314
  win        count: 1 | tf-idf: 0.314
```

**Task 7 table for D2:**

| Word | Count | TF-IDF weight |
|---|---:|---:|
| call | 1 | 0.389 |
| cash | 1 | 0.389 |
| entry | 1 | 0.389 |
| free | 1 | 0.314 |
| now | 1 | 0.314 |
| prizes | 1 | 0.389 |
| today | 1 | 0.314 |
| win | 1 | 0.314 |

**Answer.** In D2 every word appears once, so the difference in weights comes entirely from document frequency: `call`, `cash`, `entry` and `prizes` occur only in D2, so their IDF is higher, while `free`, `now`, `today` and `win` occur in two documents, so their IDF is lower. In D1, `win` gets the highest weight (0.574) because it appears twice, which doubles its term frequency, and it still has a relatively high IDF because it occurs in only two documents. The other D1 words occur once, so they receive 0.356 or 0.287 depending on their document frequency.

### Task 8 – TF-IDF by hand

```text
=== 8) TF-IDF BY HAND FOR D2 ===
D2: Free entry: win cash prizes today. Call now.
D2 tokens: ['free', 'entry', 'win', 'cash', 'prizes', 'today', 'call', 'now']
Number of tokens in D2: 8
N (number of documents): 5

  cash:
    count in D2 = 1
    df = 1
    TF = 1 / 8 = 0.1250
    IDF = log(5 / 1) = 1.6094
    textbook TF-IDF = 0.1250 x 1.6094 = 0.2012
    scikit-learn TF-IDF = 0.3891

  today:
    count in D2 = 1
    df = 2
    TF = 1 / 8 = 0.1250
    IDF = log(5 / 2) = 0.9163
    textbook TF-IDF = 0.1250 x 0.9163 = 0.1145
    scikit-learn TF-IDF = 0.3140
```

**Hand calculation.** D2 has 8 tokens. For `cash`, the count is 1 and the document frequency is 1 (only D2 contains `cash`):

$$
\text{TF} = \frac{1}{8} = 0.1250, \qquad
\text{IDF} = \log\frac{5}{1} = 1.6094, \qquad
\text{TF-IDF} = 0.1250 \times 1.6094 = 0.2012.
$$

For `today`, the count is 1 and the document frequency is 2 (D2 and D3):

$$
\text{TF} = \frac{1}{8} = 0.1250, \qquad
\text{IDF} = \log\frac{5}{2} = 0.9163, \qquad
\text{TF-IDF} = 0.1250 \times 0.9163 = 0.1145.
$$

**Answer.** The textbook formula ranks `cash` above `today` (0.2012 versus 0.1145), and scikit-learn ranks them in the same order (0.3891 versus 0.3140). The absolute values differ because scikit-learn smooths the IDF and L2-normalises each row, but the relative order is the same: the rarer word `cash` receives the higher weight.

### Task 9 – Sparse matrix

```text
=== 9) SPARSE MATRIX ===
Total cells:    160
Non-zero cells: 39
Zero cells:     121
Percentage of zeros: 75.62 %
```

**Answer.** The count matrix has $5 \times 32 = 160$ cells, of which 39 are non-zero and 121 are zero, so 75.62% of the matrix is zero. The non-zero count is smaller than the 41 tokens because a repeated word inside one document is still one non-zero cell: D1 has two `win` tokens but one `win` cell, and D5 has two `the` tokens but one `the` cell. Sparse storage keeps only the 39 values and their coordinates, which is much cheaper than storing all 160 cells.

### Task 10 – Fit and transform

```text
=== 10) FIT ON D1-D4, TRANSFORM N1 AND N2 ===
Training vocabulary: ['are', 'before', 'call', 'cash', 'claim', 'class', 'entry', 'for', 'free', 'lecture', 'lunch', 'me', 'meeting', 'notes', 'now', 'phone', 'please', 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today', 'we', 'win']

N1: Claim your free prize now
         are before call cash claim class entry for free lecture lunch me meeting notes now phone please prizes reply send still the to today we win
N1         0      0    0    0     1     0     0   0    1       0     0  0       0     0   1     0      0      0     0    0     0   0  0     0  0   0
Out-of-vocabulary words: ['your', 'prize']

N2: Is the lunch meeting moved to tomorrow?
         are before call cash claim class entry for free lecture lunch me meeting notes now phone please prizes reply send still the to today we win
N2         0      0    0    0     0     0     0   0    0       0     1  0       1     0   0     0      0      0     0    0     0   1  1     0  0   0
Out-of-vocabulary words: ['is', 'moved', 'tomorrow']
```

**Answer.** N1 is represented by the columns `claim`, `free` and `now`; `your` and `prize` are out of vocabulary and are ignored. N2 is represented by `lunch`, `meeting`, `the` and `to`; `is`, `moved` and `tomorrow` are ignored. `prize` is unknown even though D2 contains `prizes` because the vectorizer matches exact tokens: `prizes` and `prize` are different strings, and no stemming or lemmatisation was applied. Stemming or lemmatisation from Assignment 2 would map both forms to the same stem or lemma and remove this OOV problem.

### Task 11 – Data leakage

```text
=== 11) DATA LEAKAGE ===
Fitted on D1-D4 only: ['are', 'before', 'call', 'cash', 'claim', 'class', 'entry', 'for', 'free', 'lecture', 'lunch', 'me', 'meeting', 'notes', 'now', 'phone', 'please', 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today', 'we', 'win']
Fitted on D1-D4 + N1 + N2: ['are', 'before', 'call', 'cash', 'claim', 'class', 'entry', 'for', 'free', 'is', 'lecture', 'lunch', 'me', 'meeting', 'moved', 'notes', 'now', 'phone', 'please', 'prize', 'prizes', 'reply', 'send', 'still', 'the', 'to', 'today', 'tomorrow', 'we', 'win', 'your']
Words that leaked from the test set: ['is', 'moved', 'prize', 'tomorrow', 'your']
```

**Answer.** The training-only vocabulary has 26 words, while the vocabulary fitted on D1–D4 plus N1 and N2 has 31 words. The five extra words `is`, `moved`, `prize`, `tomorrow` and `your` come from the test texts, so they are the words that leaked. If the vectorizer were fitted on all texts before evaluation, the model would have columns for test-only words and IDF values computed with test data; the test score would look better than it would on genuinely unseen messages. The correct order is to split first, fit the vectorizer on the training texts only, and transform the test texts with that fitted vectorizer.

### Task 12 – No meaning

```text
=== 12) NO MEANING: COUNTS AND COSINE SIMILARITY ===
Vocabulary: ['delicious', 'food', 'tasty', 'terrible', 'the', 'was']

Review vectors:
         delicious food tasty terrible the was
R1               0    1     1        0   1   1
R2               1    1     0        0   1   1
R3               0    1     0        1   1   1

Cosine similarity matrix:
           R1   R2   R3
R1       1.00 0.75 0.75
R2       0.75 1.00 0.75
R3       0.75 0.75 1.00
R1 vs R2 (tasty vs delicious): 0.75
R1 vs R3 (tasty vs terrible):  0.75
```

**Answer.** R1 is **not** more similar to R2 than to R3: both similarities are 0.75. All three reviews share `the`, `food` and `was`, so each pair has three common words out of four, and cosine similarity only sees those exact matches. `tasty`, `delicious` and `terrible` are separate, orthogonal columns; the vectorizer does not know that `tasty` and `delicious` are synonyms, or that `terrible` is the opposite of `tasty`. This is the main limitation of Bag of Words and TF-IDF: they represent word identity and frequency, not meaning.

### Which vectorizer settings would I use for a spam detector?

For a phone spam detector I would start with **TF-IDF features, lowercasing, unigrams plus bigrams, `min_df=2`, and a generous `max_features` cap** (for example 20,000–50,000), and I would **avoid aggressive stopword removal**. The practical output shows why. TF-IDF gives rare, discriminative words a higher weight: in D2, `cash` and `prizes` receive 0.389 while the more common `today` receives 0.314, and in D1 the repeated `win` receives 0.574. That is useful for spam, where rare words such as `claim`, `prize` and `cash` are strong cues. Bigrams such as `not good` show that local order can preserve meaning that unigrams lose, and spam phrases such as `call now` or `free entry` are naturally bigrams. I would avoid a small `max_features` because the `max_features=5` run kept only `free`, `lecture`, `now`, `the` and `win`, dropping informative words such as `cash`, `prizes` and `claim`; a cap is useful for memory, but it should be large enough to keep the useful vocabulary. I would also avoid `stop_words="english"` in its default form, because it removed `now` and `not` from this corpus, and `not` is exactly the kind of word that can reverse a message's meaning; a small custom stop list or no stop list is safer for short SMS. Finally, I would keep counts plus Multinomial Naive Bayes as a cheap baseline and compare it with TF-IDF plus a linear model on validation data. The review experiment is a warning that neither representation understands synonyms or paraphrases: `tasty`, `delicious` and `terrible` all have similarity 0.75 because the shared function words dominate. For a real filter I would add character n-grams or subword features to catch obfuscated spam such as `fr33` and `pr1ze`, and I would never fit the vectorizer on the test set, because the leakage example added five test-only words to the vocabulary and would make the evaluation optimistic.

## References

1. Aggarwal, C. C., & Zhai, C. (Eds.). (2012). *Mining text data*. Springer. https://doi.org/10.1007/978-1-4614-3223-4
2. Ali, M. M. (2026a). Resource-efficient Somali news classification using TF-IDF and linear SVM: A supervised baseline on a curated news corpus. *Scientific Reports*. https://doi.org/10.1038/s41598-026-68159-5
3. Ali, M. M. (2026b). Resource-efficient Somali news classification: A supervised linear SVM approach on a novel text corpus [Preprint]. *Research Square*. https://doi.org/10.21203/rs.3.rs-9881666/v1
4. Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python*. O'Reilly Media.
5. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research, 3*, 993–1022.
6. Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science, 41*(6), 391–407. https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9
7. Forman, G. (2003). An extensive empirical study of feature selection metrics for text classification. *Journal of Machine Learning Research, 3*, 1289–1305.
8. Fox, C. (1990). A stop list for general text. *ACM SIGIR Forum, 24*(1–2), 19–21. https://doi.org/10.1145/378881.378888
9. Harris, Z. S. (1954). Distributional structure. *Word, 10*(2–3), 146–162. https://doi.org/10.1080/00437956.1954.11659520
10. Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In *Proceedings of the 10th European Conference on Machine Learning* (pp. 137–142). Springer. https://doi.org/10.1007/BFb0026683
11. Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
12. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), 1–21. https://doi.org/10.1145/2382577.2382579
13. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In *Proceedings of EMNLP 2018: System Demonstrations* (pp. 66–71). https://doi.org/10.18653/v1/D18-2012
14. Lovins, J. B. (1968). Development of a stemming algorithm. *Mechanical Translation and Computational Linguistics, 11*(1–2), 22–31.
15. Luhn, H. P. (1958). The automatic creation of literature abstracts. *IBM Journal of Research and Development, 2*(2), 159–165. https://doi.org/10.1147/rd.22.0159
16. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.
17. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems 26* (pp. 3111–3119).
18. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … VanderPlas, J. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
19. Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137. https://doi.org/10.1108/eb046814
20. Robertson, S. E., & Sparck Jones, K. (1976). Relevance weighting of search terms. *Journal of the American Society for Information Science, 27*(3), 129–146. https://doi.org/10.1002/asi.4630270302
21. Salton, G., & McGill, M. J. (1983). *Introduction to modern information retrieval*. McGraw-Hill.
22. scikit-learn developers. (2024). *Working with text data: Feature extraction*. https://scikit-learn.org/stable/modules/feature_extraction.html
23. Sebastiani, F. (2002). Machine learning in automated text categorization. *ACM Computing Surveys, 34*(1), 1–47. https://doi.org/10.1145/505282.505283
24. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. In *Proceedings of ACL 2016* (pp. 1715–1725). https://doi.org/10.18653/v1/P16-1162
25. Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526
26. Wilbur, W. J., & Sirotkin, K. (1992). The automatic identification of stop words. *Journal of Information Science, 18*(1), 45–55. https://doi.org/10.1177/016555159201800106
27. Yang, Y., & Pedersen, J. O. (1997). A comparative study on feature selection in text categorization. In *Proceedings of the 14th International Conference on Machine Learning* (pp. 412–420). Morgan Kaufmann.
