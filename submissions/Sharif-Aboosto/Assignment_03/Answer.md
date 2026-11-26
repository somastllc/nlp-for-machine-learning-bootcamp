# Assignment 3 – Vectorization

## Part A – Research

### 1. Text vectorization, fixed-length features and one-hot encoding

#### 1.1 What is text vectorization?

Text vectorization is the step that turns raw text into vectors of numbers. Each document becomes a point in a numeric space, and a learning algorithm can then measure, compare and separate those points. In information retrieval this idea is called the vector space model (Salton, Wong & Yang, 1975): every document and every query is a vector, and each dimension stands for one term of the vocabulary. Jurafsky and Martin (2025, ch. 5–6) describe the same idea for NLP in general. Vectors are simply the way a computer can "hold" words and documents so that it can do arithmetic with them.

#### 1.2 Why every document needs the same number of features

Most classic ML models (logistic regression, Naive Bayes, SVMs, decision trees, k-NN) learn from a design matrix X of shape (n_samples, n_features). Every row has to have the same length because:

a linear model learns one weight per feature (w·x + b); a row with an extra column would have no weight, and a missing column would leave a weight with nothing to multiply;
distance and similarity functions (Euclidean distance, cosine similarity) are only defined between vectors of the same dimension;
column j must mean the same thing in every row (for example "count of the word free"); otherwise the model cannot learn that feature j is a spam signal.

Texts, however, have different lengths: an SMS may have 5 words and an article 2,000. Vectorization solves this by fixing the dimensions in advance (one per vocabulary word) so that every document, short or long, is mapped to a vector of length |V|.

#### 1.3 One-hot encoding of words

In one-hot encoding, every word in a vocabulary of size |V| gets an index. A word is represented by a vector of length |V| that is all zeros except for a single 1 at that word's index. With the vocabulary [cash, class, free, lunch, notes, win]:

word	vector
cash	[1, 0, 0, 0, 0, 0]
free	[0, 0, 1, 0, 0, 0]
win	[0, 0, 0, 0, 0, 1]

One-hot encoding is a good encoding for one word, but it is not enough for a whole document:

Variable size. A sentence of k words becomes k vectors (a k × |V| matrix). Two sentences of different length produce matrices of different shape, which breaks the fixed-length requirement above.
No similarity. Every pair of different one-hot vectors is orthogonal (dot product 0), so good is as far from great as it is from terrible (Jurafsky & Martin, 2025, ch. 6).
Size. With a realistic vocabulary of 50,000 words, each word vector has 49,999 zeros.

To get one fixed-length vector per document we must combine the word vectors, for example by adding them. Summing one-hot vectors is exactly what Bag of Words does (Section 2).

Real-life example. A mobile-money company in Mogadishu wants to flag fraudulent SMS messages. The messages "Win cash" (2 words) and "Congratulations, you won cash, reply now to claim" (8 words) would become a 2 × |V| and an 8 × |V| matrix under word-level one-hot encoding. A logistic-regression classifier cannot take both as inputs. If we add up the one-hot vectors for each message, both become a single vector of length |V|, and the classifier can learn one weight for "cash", one for "win", and so on.

### 2. Bag of Words and the document-term matrix

#### 2.1 Bag of Words (BoW)

The Bag of Words model represents a document by which vocabulary words it contains and how often, ignoring grammar and word order. The steps are:

Tokenize every document (split into words, usually after lowercasing).
Build a vocabulary of all distinct tokens in the training corpus and give each one a column index.
Count: for every document, store how many times each vocabulary word occurs.

The idea goes back to Harris (1954), who discussed language as a distribution of elements, and became standard in IR through Salton's SMART system (Salton et al., 1975; Manning, Raghavan & Schütze, 2008, ch. 6).

#### 2.2 The document-term matrix (DTM)

The result is a document-term matrix:

part	meaning
Row	one document (D1, D2, …)
Column	one term (word) of the vocabulary
Cell (i, j)	a weight for term j in document i: the raw count in BoW, or a binary 0/1 value, or a TF-IDF weight

(Its transpose, with terms as rows, is called a term-document matrix; Manning et al., 2008, §1.1.)

#### 2.3 Why a "bag"?

Imagine writing each word of a sentence on a slip of paper and throwing all the slips into a bag. You can still count how many slips say "free", but you no longer know which slip came first. "Dog bites man" and "man bites dog" produce the same bag, and therefore the same vector.

#### 2.4 When losing order matters little vs. a lot

Little harm – topic classification / spam filtering. Whether a news article is about sport or politics depends mostly on which words occur (goal, striker, league vs. parliament, election, minister). Classic studies found BoW with a linear classifier very strong for topic categorization (Joachims, 1998).
Real problems – sentiment with negation or contrast, and anything about who-did-what. "The lecture was not good, but the notes were great" and "The lecture was good, but the notes were not great" contain exactly the same words, yet the opinion about the lecture is reversed. Pang, Lee and Vaithyanathan (2002) showed that sentiment is harder than topic classification for BoW models and had to add special negation handling. Machine translation and question answering ("Who paid whom?") also need word order.

### 3. Vocabulary size and sparsity

#### 3.1 Why the matrix becomes huge and mostly zero

The vocabulary size |V| is the number of distinct terms kept as columns. In a real corpus it grows with every new document: new names, spelling variants, typos, numbers and rare words keep appearing. Heaps' law describes this as roughly V = k·T^β with β ≈ 0.4–0.6, where T is the number of tokens (Manning et al., 2008, §5.1.1). At the same time, Zipf's law says that a few words are very frequent while most words are rare.

Together this means a corpus can have hundreds of thousands of columns, but any one document uses only a few dozen or a few hundred of them. Most cells are zero, so the matrix is sparse. Even our tiny 5-document corpus is about 76% zeros (Part B, task 9); real corpora are often more than 99% zeros.

#### 3.2 Storing sparse matrices

Storing every zero wastes memory: 100,000 documents × 100,000 words × 8 bytes = 80 GB. Sparse formats store only the non-zero cells. scikit-learn's vectorizers return a SciPy CSR (Compressed Sparse Row) matrix with three arrays (SciPy, 2024; scikit-learn, 2024):

array	content
data	the non-zero values
indices	the column index of each value
indptr	where each row starts and ends inside data

Memory then grows with the number of non-zeros, not with rows × columns. CSR is also fast for the row-by-row operations ML models need. Search engines use a similar idea, the inverted index, which stores for each term only the list of documents that contain it (Manning et al., 2008, ch. 1–2).

#### 3.3 Settings that change the vocabulary
setting (scikit-learn)	effect
min_df	drops terms appearing in fewer than min_df documents (an integer = number of documents, a float = proportion). Removes rare words, typos and names.
max_df	drops terms appearing in more than max_df documents (e.g. 0.9). Removes corpus-specific stopwords that appear almost everywhere.
max_features	keeps only the k terms with the highest total frequency in the corpus.
lowercase (default True)	merges Win, WIN and win into one term.
stop_words="english"	removes a fixed list of function words (the, to, a, not, …). scikit-learn's own documentation warns that this list has known problems and is not suitable for every task.
token_pattern (default (?u)\b\w\w+\b)	only tokens of 2 or more alphanumeric characters are kept, so punctuation and one-letter words like a or I disappear.
stemming / lemmatization (applied through a custom tokenizer/analyzer)	reduces prizes → prize, meeting → meet, merging related forms (Porter, 1980).

Every step that merges forms (lowercasing, stemming) or removes tokens (stopwords, min_df, max_df, max_features) makes the vocabulary smaller and the matrix less sparse, but each one can also throw away useful information. For example, capital letters can be a spam signal and not matters for sentiment.

### 4. N-grams

An n-gram is a sequence of n consecutive tokens (Jurafsky & Martin, 2025, ch. 3):

n	name	example from "the lecture was not good"
1	unigram	the, lecture, was, not, good
2	bigram	the lecture, lecture was, was not, not good
3	trigram	the lecture was, lecture was not, was not good

A "bag of n-grams" still ignores the global order of the document, but each bigram or trigram keeps local order inside a small window of 2 or 3 words. That is enough to capture negations (not good), compound names (New York, machine learning) and fixed phrases (free entry, call now).

Example where a bigram changes the meaning. In a unigram model, "The food was not bad" contains bad and looks negative. With bigrams, the feature not bad appears and can be learned as a positive (or neutral) signal. In our corpus, the bigram not good in D5 keeps the reviewer's opinion about the lecture, while unigrams only give good and not as two unrelated features.

Cost of larger n.

Vocabulary explosion: a document of L tokens has up to L − n + 1 n-grams of each size. Most of them are unique, so the vocabulary grows quickly (32 → 67 → 98 features in our small corpus, task 6).
More sparsity: most trigrams appear only once, so they help little in generalizing to new texts and increase the risk of overfitting.
Memory and speed: more columns mean more memory and slower training.

For this reason, practitioners usually use ngram_range=(1, 2) and combine it with min_df to remove rare n-grams. Zhang, Zhao and LeCun (2015), for example, limited their bag-of-n-grams baseline to the 500,000 most frequent n-grams.

### 5. TF-IDF

#### 5.1 Idea

Raw counts give too much weight to words that are frequent everywhere (the, now, today). TF-IDF (term frequency – inverse document frequency) multiplies two factors:

TF: how important the word is inside this document (it appears often here);
IDF: how specific the word is to only a few documents in the collection.

A word gets a high TF-IDF weight when it is frequent in this document but rare in the corpus. That is the kind of word that tells documents apart.

#### 5.2 Origin of IDF

IDF was introduced by Karen Spärck Jones (1972) in "A statistical interpretation of term specificity and its application in retrieval" (Journal of Documentation). She argued that a term's specificity should be measured statistically, by the number of documents it is assigned to: a term used in many documents matches too many documents to be useful for finding the right ones. Her weighting (roughly log(N/n) + 1) greatly improved retrieval. Salton and Buckley (1988) later compared many term-weighting schemes and established TF × IDF as the standard. Robertson (2004) gave a theoretical justification based on probability and information theory.

#### 5.3 Textbook formula

For term t, document d and a collection of N documents (Manning et al., 2008, §6.2):

tf(t, d)     = count(t, d)                 (often normalized: count(t, d) / |d|)
idf(t)       = log( N / df(t) )            df(t) = number of documents containing t
tfidf(t, d)  = tf(t, d) × idf(t)

Why rare words get higher weights. N / df(t) is large when df(t) is small. A word in 1 of 5 documents gets log(5) ≈ 1.61, a word in all 5 gets log(1) = 0. The logarithm reduces the effect, so a word in 1 of 1,000,000 documents is not a million times more important than a common one. A word that occurs in every document cannot distinguish any document from another, so it gets weight 0.

#### 5.4 How scikit-learn's TfidfVectorizer differs
aspect	textbook	scikit-learn default
TF	count or count/length	raw count (sublinear_tf=True gives 1 + log(tf))
IDF	log(N/df)	ln((1 + N) / (1 + df)) + 1 (smooth_idf=True)
Normalization	usually none	L2-normalization of every row (norm="l2")
Smoothing (+1 inside): acts as if one extra document containing every term exists, which prevents division by zero.
+1 outside: a word that appears in every document gets weight 1 instead of 0, so it is weakened but not deleted completely.
L2 normalization: every document vector is divided by its Euclidean length, so all rows have length 1. Long documents no longer get large values just because they are long, and the dot product of two rows equals their cosine similarity.

(scikit-learn, 2024, §6.2.3.4 "Tf–idf term weighting".)

#### 5.5 Weights, not probabilities

TF-IDF values are weights that show relative importance; they are not probabilities:

they do not sum to 1 (after L2 normalization, the sum of squares is 1);
raw TF-IDF values can be larger than 1 (e.g. a count of 5 × idf 2.1 = 10.5);
they do not describe any random event, such as "the probability that the next word is cash".

So a TF-IDF of 0.57 does not mean "57% chance"; it only means this word is relatively more important for this document than a word with 0.29. Probabilistic models such as multinomial Naive Bayes or language models estimate real probabilities like P(word | class), which is a different thing (Jurafsky & Martin, 2025, ch. 4).

### 6. fit, transform, fit_transform, OOV words and data leakage

#### 6.1 The three methods

method	what it does
fit(texts)	learns parameters from the texts: the vocabulary (word → column) and, for TF-IDF, the IDF of every word. Produces no matrix.
transform(texts)	applies the learned vocabulary and IDF to (possibly new) texts and returns a matrix with exactly the same columns. Learns nothing.
fit_transform(texts)	does fit and then transform on the same texts in one (more efficient) step. Use it only on training data.

This follows scikit-learn's general estimator API, where fit learns state from data and transform uses that state (Buitinck et al., 2013).

#### 6.2 Out-of-vocabulary (OOV) words

When transform meets a word that was not in the training vocabulary, there is no column for it, so the word is silently ignored. No error is raised and the information is lost. If every word of a new text is OOV, its vector is all zeros. This is a real weakness for languages with rich morphology, like Somali, where one root can appear in many forms. Common fixes are stemming/lemmatization, character n-grams (analyzer="char_wb"), a HashingVectorizer, or subword models.

#### 6.3 Data leakage

Data leakage happens when information from the test set (or from the future) is used while building the model (Kaufman et al., 2012). In vectorization, leakage happens when the vectorizer is fit on the whole dataset before splitting into train and test:

the vocabulary then contains words that only appear in the test set, which would be unknown OOV words in real use;
the IDF values are computed with test documents included, so the training features already "know" the test set's word statistics;
min_df, max_df and max_features choose features using test data.

Why results look better than they really are. The test set is supposed to simulate unseen data. With leakage, the model is evaluated on data that has partly been "seen" through the features, so the test score measures performance in an easier, unrealistic situation. When the model is deployed, real new texts will have OOV words and different statistics, and accuracy drops. Hastie, Tibshirani and Friedman (2009, §7.10.2) call doing feature selection before cross-validation "the wrong way" to cross-validate for the same reason.

Correct order:

Split the data into train and test (or set up cross-validation folds) first.
fit (or fit_transform) the vectorizer on the training set only.
transform the test set with the already fitted vectorizer.
Train the model on the training matrix and evaluate it on the test matrix.

The safest way is to put the vectorizer and the model inside a scikit-learn Pipeline, so that cross-validation automatically refits the vectorizer on every training fold only.

### 7. Case study: news classification in isiZulu and Siswati

Paper. Madodonga, A., Marivate, V., & Adendorff, M. (2023). Izindaba-Tindzaba: Machine learning news categorisation for long and short text for isiZulu and Siswati. Journal of the Digital Humanities Association of Southern Africa, 4(1). arXiv:2306.07426.

(I searched for a Somali study first. The closest one I found, Mohamed et al. (2025), builds a Somali transformer model, SomBERTa, for fake-news and toxicity detection, but it compares language models rather than BoW/TF-IDF. So I chose an African-language study that compares exactly the representations from this assignment.)

Task. Topic classification of news (long articles and short headlines) into IPTC news categories, for two low-resource South African languages.

Data.

dataset	source	size after cleaning	categories
isiZulu articles + titles	Isolezwe newspaper	563 items	5 (crime/law, economy, education, politics, society)
Siswati headlines	SABC Ligwalagwala FM Facebook page	68 items	5

Word2vec embeddings were trained on separate unlabelled corpora (about 5 million isiZulu tokens and 0.5 million Siswati tokens). The authors also tested two ways of dealing with the tiny, imbalanced data: data augmentation (replacing words with similar words found by Word2vec) and SMOTE oversampling.

Representations compared. Bag of Words (counts) and TF-IDF (both limited to 20,000 features), and Word2vec embeddings (300 dimensions). Classifiers: Logistic Regression, Naive Bayes, XGBoost and an LSTM (with Word2vec only). Evaluation used 5-fold cross-validation and F1-score.

#### Results (best F1, %).

dataset	setting	best Word2vec	best BoW / TF-IDF
isiZulu articles	original	82.8 (LSTM)	36.1 (LR + BoW)
isiZulu articles	augmented	95.2 (XGBoost)	83.1 (LR + BoW)
isiZulu titles	original	72.0 (LSTM)	17.6 (LR + TF-IDF)
Siswati titles	original	81.1 (LSTM)	26.3 (LR + BoW)
Siswati titles	augmented	93.2 (LSTM)	75.6 (LR + TF-IDF)

Word2vec beat BoW and TF-IDF on every dataset, and the gap was largest on short headlines with little data.

What it shows about meaning. The authors report the gap but do not test why it happens; they mention possible reasons and leave a deeper analysis to future work. The results, however, fit the main limitation of count-based features. BoW and TF-IDF treat every word as an unrelated column. A headline that uses a synonym or a different inflected form of a word seen in training (very common in agglutinative languages like isiZulu and Siswati) shares no features with the training examples, so the classifier has nothing to go on. With only 68–563 labelled texts, most words are seen once or never. Word2vec, trained on millions of unlabelled tokens, puts words that occur in similar contexts close together (the distributional hypothesis; Harris, 1954; Mikolov et al., 2013), so a new but related word still produces a useful vector. The success of the augmentation method, which works by swapping in semantically similar words, makes the same point: the BoW/TF-IDF models did much better once the data was changed to contain those meaning relations that the representation itself cannot see.

## Part B – Practical

All numbers below come from output.txt, produced by python vectorization_practice.py > output.txt (scikit-learn 1.9.1, NumPy 2.5.3).

### Task 1 – One-hot encoding
free cash      ->  free: [0,0,1,0,0,0]   cash: [1,0,0,0,0,0]                       (2 vectors)
win free cash  ->  win: [0,0,0,0,0,1]   free: [0,0,1,0,0,0]   cash: [1,0,0,0,0,0] (3 vectors)

Answer. One-hot encoding produces one vector per word, so a 2-word sentence gives 2 vectors and a 3-word sentence gives 3. Each vector always has length 6 (the vocabulary size), but the number of vectors depends on the sentence length. Sentences of different length therefore get representations of different shape (2×6 vs 3×6), and a standard model that needs a fixed number of features cannot use them directly.

### Task 2 – Bag of Words

Vocabulary: 32 words, shape (5, 32).

Document-term matrix

doc	are	before	but	call	cash	claim	class	entry	for	free	good	great	lecture	lunch	me	meeting	not	notes	now	phone	please	prizes	reply	send	still	the	to	today	was	we	were	win
D1	0	0	0	0	0	1	0	0	0	1	0	0	0	0	0	0	0	0	1	1	0	0	1	0	0	0	1	0	0	0	0	2
D2	0	0	0	1	1	0	0	1	0	1	0	0	0	0	0	0	0	0	1	0	0	1	0	0	0	0	0	1	0	0	0	1
D3	1	0	0	0	0	0	0	0	1	0	0	0	0	1	0	1	0	0	0	0	0	0	0	0	1	0	0	1	0	1	0	0
D4	0	1	0	0	0	0	1	0	0	0	0	0	1	0	1	0	0	1	0	0	1	0	0	1	0	1	0	0	0	0	0	0
D5	0	0	1	0	0	0	0	0	0	0	1	1	1	0	0	0	1	1	0	0	0	0	0	0	0	2	0	0	1	0	1	0

Answer. In D1 the cell win = 2, because D1 contains "Win" and "WIN". The vectorizer lowercases both to win before counting. (D5 also has a 2, for the.) The vectors for win free phone and phone free win are identical (the script prints True), with 1s in the free, phone and win columns. Bag of Words only records which words appear and how often, not their order.

### Task 3 – Tokens vs vocabulary size

The corpus has 41 tokens but only 32 distinct words. Words appearing in more than one document (all with df = 2): free, now, win (D1, D2), today (D2, D3), lecture, notes, the (D4, D5). The other 25 words appear in only one document. Note that a is not counted as a token, because the default token pattern needs at least 2 characters.

### Task 4 – Preprocessing inside the vectorizer

setting (fit on D1)	vocabulary
default	claim, free, now, phone, reply, to, win (7)
lowercase=False	FREE, Reply, WIN, Win, claim, now, phone, to (8)
stop_words="english"	claim, free, phone, reply, win (5)

Answer. a disappeared in every setting because the default token_pattern only keeps tokens of 2+ characters. ! disappeared because punctuation is never part of a token. With the default lowercasing, Win and WIN became one word, win, with count 2. With lowercase=False they stayed two separate columns (WIN, Win), and FREE and Reply kept their capitals. The stopword list removed now and to.

Fit on D1–D5, stop_words="english" removed not (along with the, was, but, were, …). D5 becomes ['lecture', 'good', 'notes', 'great']. It now looks fully positive: the negation is lost, and "the lecture was not good" is represented as if the lecture was good.

### Task 5 – Limiting the vocabulary

setting	size	vocabulary
full	32	(all words)
min_df=2	7	free, lecture, notes, now, the, today, win
max_features=5	5	free, lecture, now, the, win

Answer. Only frequent words survived: the 7 words that appear in at least two documents, or the 5 with the highest total counts (the and win have 3; the rest have 2, with ties broken by the vectorizer). These are a mix of useful spam words (free, win, now) and a useless function word (the). Very specific words such as cash, prizes, claim and lunch were removed even though they are informative. On a tiny corpus these limits are too strong; on a large corpus min_df mostly removes typos and rare names.

### Task 6 – N-grams

D5 with (1, 1): but, good, great, lecture, not, notes, the, was, were (9)
D5 with (1, 2): adds but the, good but, lecture was, not good, notes were, the lecture, the notes, was not, were great (18)
ngram_range on D1–D5	vocabulary size
(1, 1)	32
(1, 2)	67 (×2.09)
(1, 3)	98 (×1.46)

Answer. The bigram not good keeps the reviewer's opinion: the lecture was bad. With unigrams, not and good are separate and the model cannot tell "not good" from "good". Adding bigrams roughly doubled the vocabulary, and adding trigrams tripled it compared to unigrams (32 → 98), although the corpus has only 41 tokens. Almost every new n-gram appears only once, so the extra columns are mostly sparse, one-off features.

### Task 7 – TF-IDF

TF-IDF matrix (rounded):

doc	are	before	but	call	cash	claim	class	entry	for	free	good	great	lecture	lunch	me	meeting	not	notes	now	phone	please	prizes	reply	send	still	the	to	today	was	we	were	win
D1	0	0	0	0	0	.36	0	0	0	.29	0	0	0	0	0	0	0	0	.29	.36	0	0	.36	0	0	0	.36	0	0	0	0	.57
D2	0	0	0	.39	.39	0	0	.39	0	.31	0	0	0	0	0	0	0	0	.31	0	0	.39	0	0	0	0	0	.31	0	0	0	.31
D3	.39	0	0	0	0	0	0	0	.39	0	0	0	0	.39	0	.39	0	0	0	0	0	0	0	0	.39	0	0	.31	0	.39	0	0
D4	0	.38	0	0	0	0	.38	0	0	0	0	0	.31	0	.38	0	0	.31	0	0	.38	0	0	.38	0	.31	0	0	0	0	0	0
D5	0	0	.32	0	0	0	0	0	0	0	.32	.32	.26	0	0	0	.32	.26	0	0	0	0	0	0	0	.51	0	0	.32	0	.32	0

#### D2: count vs TF-IDF

word	count	df	idf (sklearn)	TF-IDF
call	1	1	2.0986	0.39
cash	1	1	2.0986	0.39
entry	1	1	2.0986	0.39
prizes	1	1	2.0986	0.39
free	1	2	1.6931	0.31
now	1	2	1.6931	0.31
today	1	2	1.6931	0.31
win	1	2	1.6931	0.31

D1: count vs TF-IDF: win 2 → 0.57; claim, phone, reply, to 1 → 0.36; free, now 1 → 0.29.

Answer. In D2 every count is 1, so the difference comes only from IDF. call, cash, entry, prizes appear only in D2 (df = 1, idf 2.10), while free, now, today, win also appear in another document (df = 2, idf 1.69), so they are less specific to D2 and get less weight. In D1, win has idf 1.69, lower than claim (2.10), but it appears twice: 2 × 1.69 = 3.39 is bigger than 1 × 2.10. Its higher term frequency is more important than its lower IDF, so win gets the top weight. Both rows have L2 norm 1.0, which shows the normalization.

### Task 8 – TF-IDF by hand (D2)

D2 as the vectorizer counts it: free, entry, win, cash, prizes, today, call, now → 8 words; N = 5.

word	count	TF = count/8	df	IDF = ln(5/df)	TF × IDF	sklearn (task 7)
cash	1	0.1250	1 (D2)	ln 5 = 1.6094	0.2012	0.3891
today	1	0.1250	2 (D2, D3)	ln 2.5 = 0.9163	0.1145	0.3140

Answer. Yes, both methods rank cash above today, because the TF is equal and cash has the lower document frequency. The actual numbers are different: scikit-learn uses smoothed IDF (ln(6/2)+1 = 2.0986 and ln(6/3)+1 = 1.6931), raw counts instead of count/length, and L2 normalization. The +1 also reduces the gap: in the textbook formula cash is 1.76× today, in scikit-learn only 1.24×.

### Task 9 – Sparse matrix

total cells	non-zero	zero	% zeros
160 (5 × 32)	39	121	75.62%

The matrix is a csr_matrix that stores only the 39 non-zero values. Even five short sentences make a matrix that is three-quarters empty; a real corpus is usually over 99% zeros.

### Task 10 – Fit and transform

Training vocabulary (D1–D4): 26 words.

text	known words (columns = 1)	OOV words (ignored)
N1 "Claim your free prize now"	claim, free, now	your, prize
N2 "Is the lunch meeting moved to tomorrow?"	the, lunch, meeting, to	is, moved, tomorrow

Answer. prize is unknown because the vectorizer compares exact strings: prizes was learned, but prize is a different string with no column, so it is dropped. This loses an important spam word. Stemming or lemmatization (from Assignment 2) would map both to prize. My script shows this with a simple plural-stripping stemmer: prize enters the vocabulary, and N1 then matches 4 known words instead of 3.

### Task 11 – Data leakage

Fit on D1–D4: 26 words.
Fit on D1–D4 + N1 + N2: 31 words.
Leaked words: is, moved, prize, tomorrow, your.

Answer. These five words exist only in the test texts, but fitting on train + test gave them columns. The model is then evaluated on features it could never have in real use, where they would be OOV. The script also shows that leakage changes the IDF of the training words: win goes from 1.51 to 1.85, and today from 1.51 to 1.85. So even the training features are affected by the test set, and the test score becomes too optimistic.

### Task 12 – No meaning

Vocabulary: delicious, food, tasty, terrible, the, was

	delicious	food	tasty	terrible	the	was
R1	0	1	1	0	1	1
R2	1	1	0	0	1	1
R3	0	1	0	1	1	1

#### Cosine similarity:

	R1	R2	R3
R1	1.00	0.75	0.75
R2	0.75	1.00	0.75
R3	0.75	0.75	1.00

Answer. No. R1 (tasty) is exactly as similar to R3 (terrible) as to R2 (delicious): 0.75 in both cases. Every pair shares the same 3 words (the, food, was) and differs in one word. For Bag of Words, tasty, delicious and terrible are just three different columns with no relation between them. A human sees that R1 and R2 mean the same thing and R3 means the opposite, but the count vectors cannot show this. Capturing it needs representations learned from context, such as word embeddings.

#### Which settings for a spam detector?

For a spam detector I would use a TfidfVectorizer with ngram_range=(1, 2), sublinear TF, and a moderate min_df (for example 2 on a real corpus), fit only on the training messages inside a Pipeline. I would add a stemmer or character n-grams. TF-IDF helps because, as task 7 showed, it raises words specific to a message (cash, prizes, claim) above words found everywhere, and win still stands out when it is repeated. Bigrams like free entry, call now and win cash are strong spam phrases. Task 10 showed that without stemming, prize was lost because only prizes was in the training data. I would avoid the built-in English stopword list: in task 4 it removed now, a typical "act now" spam word (and not, which matters for other tasks). I would also avoid a very small max_features or a high min_df on small data, because task 5 kept the but dropped cash, prizes and claim. I would keep lowercase=False in mind as an extra feature, since task 4 showed that it keeps FREE and WIN, and shouting in capitals is a known spam signal. I would not use n-grams above 2: task 6 showed the vocabulary tripling (32 → 98) with mostly one-off features. Finally, task 11 shows why the vectorizer must never be fit on test messages.

## References
Buitinck, L., Louppe, G., Blondel, M., et al. (2013). API design for machine learning software: experiences from the scikit-learn project. ECML PKDD Workshop: Languages for Data Mining and Machine Learning, 108–122. arXiv:1309.0238.
Harris, Z. S. (1954). Distributional structure. Word, 10(2–3), 146–162.
Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.), §7.10.2. Springer.
Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. Proceedings of ECML-98, 137–142.
Jurafsky, D., & Martin, J. H. (2025). Speech and Language Processing (3rd ed. draft), chapters 3 (N-gram language models), 4 (Naive Bayes and sentiment), 6 (Vector semantics and embeddings). https://web.stanford.edu/~jurafsky/slp3/
Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4), 1–21.
Madodonga, A., Marivate, V., & Adendorff, M. (2023). Izindaba-Tindzaba: Machine learning news categorisation for long and short text for isiZulu and Siswati. Journal of the Digital Humanities Association of Southern Africa, 4(1). arXiv:2306.07426. https://arxiv.org/abs/2306.07426
Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press. Chapters 1, 2, 5, 6. https://nlp.stanford.edu/IR-book/
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv:1301.3781.
Mohamed, M. A., et al. (2025). Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models. arXiv:2503.18117. https://arxiv.org/abs/2503.18117
Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. Proceedings of EMNLP 2002, 79–86.
Porter, M. F. (1980). An algorithm for suffix stripping. Program, 14(3), 130–137.
Robertson, S. (2004). Understanding inverse document frequency: On theoretical arguments for IDF. Journal of Documentation, 60(5), 503–520.
Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513–523.
Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. Communications of the ACM, 18(11), 613–620.
scikit-learn developers (2024). User Guide §6.2: Feature extraction – text feature extraction (CountVectorizer, TfidfVectorizer, stop-word warnings, tf-idf term weighting). https://scikit-learn.org/stable/modules/feature_extraction.html
SciPy developers (2024). scipy.sparse.csr_matrix documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. Journal of Documentation, 28(1), 11–21.
Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. Advances in Neural Information Processing Systems 28. arXiv:1509.01626.