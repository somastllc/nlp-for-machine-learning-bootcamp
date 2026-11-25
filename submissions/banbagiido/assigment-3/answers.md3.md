# Assignment 3 – Vectorization

**Course:** Natural Language Processing  |  **Topic:** Turning text into numbers
**Tools used for Part B:** Python 3.12, scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1

> **How to read this report.** Part A is a research essay built from books, papers and library documentation (all listed in the References section). Part B is the practical work; every number in Part B comes from my own script `vectorization_practice.py`, whose complete printed output is saved in `output.txt`.

---

# Part A – Research

## 1. Text vectorization, fixed-length features and one-hot encoding

**Definition.** *Text vectorization* is the process of mapping a piece of raw text (a document) to a numerical vector so that a mathematical model can process it. Jurafsky and Martin (2025) describe this as the step that turns language into features a classifier can weight; in the vector-space tradition it goes back to Salton, Wong and Yang (1975), who represented every document as a point in a space whose axes are index terms.

**Why every document needs the same number of features.** Most classical Machine Learning models (logistic regression, SVMs, Naive Bayes, decision trees, neural-network input layers) learn *one parameter per feature position*. A logistic regression with *d* features learns a weight vector **w** ∈ ℝ^d and computes **w · x**; the dot product is only defined when **x** also has *d* entries. Even more important, position *j* must mean the *same thing* in every row: the weight learned for column 17 during training is applied to column 17 of a new document. Text, however, arrives with variable length: a headline has six words, a news article has six hundred. Vectorization solves this by fixing one shared set of dimensions (the vocabulary) and describing every document in those dimensions, which produces the rectangular design matrix **X** (*n* documents × *d* features) that libraries such as scikit-learn expect (Pedregosa et al., 2011).

```text
raw text ──► tokenize ──► choose vocabulary (d words) ──► vector of length d per document ──► model
```

**One-hot encoding of words.** Fix a vocabulary *V* of size |V|. The one-hot vector of a word is a vector of length |V| that contains a single 1 at the word's own index and 0 everywhere else. Two properties matter:

| Property | Consequence |
|---|---|
| Length = \|V\| (often 10⁴–10⁶) | Very high-dimensional and extremely sparse |
| Any two different words are **orthogonal** (dot product 0) | Every word is equally "far" from every other word; no notion of similarity |

**Why one-hot is not enough for a whole document.** One-hot encoding describes *a word*, not a *document*. A sentence of *L* words becomes *L* vectors, i.e. an *L* × |V| matrix whose number of rows changes from sentence to sentence. A classifier needs a single fixed-size vector per document, so something must combine the word vectors (summing them gives Bag of Words, Section 2; averaging or weighting them gives other schemes). In addition, a long list of one-hot vectors keeps no information about similarity, and it grows with document length.

**Real-life example.** Imagine a news app that must sort incoming headlines into *Sports*, *Weather* and *Economy*. Suppose the vocabulary is `[rain, goal, price, flood]`.

| Headline | One-hot vectors produced |
|---|---|
| "Heavy rain causes flood" (after removing words outside the vocabulary) | `rain` → [1,0,0,0]; `flood` → [0,0,0,1] (2 vectors) |
| "Late goal wins cup final as rain falls" | `goal` → [0,1,0,0]; `rain` → [1,0,0,0] (2 vectors, different content) |
| "Price rise follows rain and flood damage" | `price`, `rain`, `flood` (3 vectors) |

The classifier cannot take "a list of 2 vectors" in one case and "a list of 3 vectors" in another. It needs, for each headline, one vector of length 4. Summing the one-hot vectors gives that single vector (for example [1,0,0,1] for the first headline), which is exactly the idea behind the Bag-of-Words model.

---

## 2. Bag of Words and the document-term matrix

**Bag of Words (BoW).** In BoW, a document is represented by *which* vocabulary words it contains and *how often*, while the order of the words is discarded (Manning, Raghavan & Schütze, 2008, ch. 6). The phrase is traditionally credited to Harris (1954), and the idea that word frequency indicates what a text is about goes back to Luhn (1957).

**The document-term matrix (DTM).** Stacking the BoW vectors of *n* documents gives a matrix:

| Element | Meaning |
|---|---|
| **Row** *i* | One document (or one text sample) |
| **Column** *j* | One vocabulary term (word or n-gram) |
| **Cell** (*i*, *j*) | The weight of term *j* in document *i*: a raw count, a 0/1 presence flag, or a TF-IDF weight |

The transposed version (terms × documents) is called the *term-document matrix* in the information-retrieval literature.

**Why a "bag"?** Think of a real bag of marbles: you can say how many red and blue marbles are inside, but not in what order they were put in. Likewise the BoW vector records *what and how many* but not *where*: "dog bites man" and "man bites dog" get exactly the same vector.

**When losing word order does little harm.** *Topic classification and spam filtering.* Words such as "goal", "match" and "league" signal sport regardless of their order, and the presence of "lottery" or "winner" is a strong spam clue by itself. Joachims (1998) showed that a linear SVM on plain word-frequency features is already very strong for topic categorization of news, because the topic is carried by *which* words occur.

**When it causes real problems.** *Sentiment analysis with negation, and any task about who did what to whom.* "The service was good, not bad" and "The service was bad, not good" contain the same words but opposite meaning. Machine translation, question answering and relation extraction are similar: the roles of words depend on order. Wang and Manning (2012) found that adding bigrams to bag-of-words features noticeably helps sentiment tasks, which is evidence that word order carries signal there.

---

## 3. Vocabulary size, sparsity and the settings that control them

**Vocabulary size** is the number of distinct terms the vectorizer keeps; it equals the number of columns *d* of the DTM. It must not be confused with the number of *tokens*, i.e. all word occurrences.

**Why real matrices become very large.** Two empirical laws explain this. *Zipf's law* (Zipf, 1949) says that a few words are extremely frequent while most words are rare. *Heaps' law* (Heaps, 1978) says that the vocabulary keeps growing with the corpus size, roughly V ≈ K·N^β with 0.4 ≲ β ≲ 0.6 for English (*N* = number of tokens). A new document nearly always brings new names, typos, numbers and rare words, so vocabularies on real collections reach tens or hundreds of thousands of columns.

**Why they are mostly zero.** A single document uses only a tiny part of the vocabulary: a 100-word message cannot contain more than 100 different words, while the vocabulary may contain 100,000. So more than 99 % of the cells in a realistic DTM are zero.

**Efficient storage.** Storing a dense matrix wastes memory: 10,000 documents × 50,000 terms = 5 × 10⁸ cells, which at 8 bytes per cell is about 4 GB. Sparse formats store *only the non-zero entries*. SciPy's **CSR** (compressed sparse row) format, which scikit-learn's vectorizers return, keeps three arrays: `data` (the non-zero values), `indices` (their column numbers) and `indptr` (where each row starts in those arrays) (SciPy documentation). If each document has about 100 non-zero terms, the 10,000-document matrix has 10⁶ non-zeros and needs roughly 12 MB (8 bytes per value plus 4 bytes per column index), which is hundreds of times smaller. Memory is proportional to the number of non-zeros, not to the number of cells.

**Controlling the vocabulary.** The table summarizes the main levers, using the names from scikit-learn's `CountVectorizer`/`TfidfVectorizer` (scikit-learn developers, n.d.).

| Setting | What it does | Typical reason | Risk |
|---|---|---|---|
| `min_df` | Drops terms found in fewer than *k* documents (or a fraction) | Removes typos, rare names; shrinks *d* a lot | Rare but *informative* words are lost |
| `max_df` | Drops terms found in more than a given number/fraction of documents | Removes corpus-specific "stop words" | May remove words that are frequent *because* they matter |
| `max_features` | Keeps only the *k* terms with the highest total frequency | Hard cap on memory | Frequency is not the same as usefulness |
| Lowercasing | Merges "Win" and "WIN" | Fewer columns, more reliable counts | Loses signals such as ALL-CAPS shouting or "US" vs "us" |
| Stop-word removal | Deletes very common function words | Fewer columns, less noise | Lists are not neutral: they can delete "not" or "no", which changes meaning (Nothman, Qin & Yurchak, 2018) |
| Stemming / lemmatization | Maps word forms to a common base ("prizes", "prize" → *priz* / *prize*) | Merges inflected forms, reduces sparsity | Porter-style stemming (Porter, 1980) can over-merge unrelated words ("university"/"universe") or produce non-words |

Note that scikit-learn's vectorizers lowercase text and use the token pattern `\b\w\w+\b` (words of at least two characters) by default, but they do **not** stem; stemming must be supplied through a custom analyzer or by pre-processing the text.

---

## 4. N-grams

An **n-gram** is a contiguous sequence of *n* tokens. A *unigram* is one word, a *bigram* two consecutive words, a *trigram* three. Using n-grams as the vocabulary terms lets a model see small *local* patterns of order, even though the overall representation is still a bag (now a "bag of n-grams").

| Text: "the court found him not guilty" | n-grams (after tokenization) |
|---|---|
| Unigrams | the, court, found, him, not, guilty |
| Bigrams | the court, court found, found him, him not, **not guilty** |
| Trigrams | the court found, court found him, found him not, him not guilty |

**A bigram that changes the meaning.** Compare "found guilty" with "found **not guilty**". With unigrams only, both texts contain the feature *guilty*, and the model cannot tell that the verdict is reversed; the feature *not* is shared with countless unrelated sentences. The bigram *not guilty* is a single feature with its own learned weight, so the verdict is represented correctly. The same happens with "not good", "no longer" or "New York" (a bigram that means a place, not "new" plus "York").

**Cost of larger n-grams.**

1. **Vocabulary explosion.** A document of *L* tokens produces about *L − n + 1* n-grams for each order *n*, and most longer n-grams occur only once, so the number of columns grows rapidly (the number of *possible* n-grams is |V|ⁿ).
2. **More sparsity and data hunger.** Each specific trigram appears in very few documents, so there are too few examples to learn a reliable weight; the model can overfit by memorizing rare phrases.
3. **Memory and time.** More columns mean larger matrices and slower training (practitioners usually combine n-grams with `min_df` or `max_features`).
4. **Diminishing returns.** Most of the gain usually comes from bigrams; trigrams and beyond help less (Wang & Manning, 2012).

---

## 5. TF-IDF

**Idea.** A word that occurs in *every* document (for example "the") tells us nothing about what makes one document different from another, while a word that occurs in only a few documents is a good *discriminator*. TF-IDF multiplies a local measure (how prominent the term is *in this document*, TF) by a global measure (how *rare* the term is across the collection, IDF).

**Where IDF came from.** Inverse document frequency was introduced by Karen Spärck Jones (1972) in a paper on *term specificity*. She observed that terms that occur in fewer documents are more specific, and that giving such terms more weight in matching improves retrieval; her weight was a logarithmic function of the number of documents containing the term. Salton and Buckley (1988) later systematized and tested term-weighting schemes and made *tf × idf* the standard weighting in the vector space model, and Robertson (2004) discussed theoretical justifications for IDF based on probabilistic retrieval.

**Textbook formula** (Manning et al., 2008):

```text
tf(t, d)    = number of times t occurs in d          (sometimes divided by the length of d)
idf(t)      = log( N / df(t) )                        N = number of documents, df(t) = documents containing t
tf-idf(t,d) = tf(t, d) × idf(t)
```

**Why rarer words get higher weights.** The fraction N/df(t) becomes larger as df(t) becomes smaller, and the logarithm turns it into a smooth, slowly growing weight. A term found in all documents has df = N, so log(N/N) = 0 and the term is switched off; a term found in one document of 1,000 has idf = log(1000) ≈ 6.9. The logarithm *dampens* the effect so that a term 1,000 times rarer is not 1,000 times more important.

**How scikit-learn differs from the textbook.** According to the documentation of `TfidfTransformer`/`TfidfVectorizer`:

| Aspect | Textbook | scikit-learn default |
|---|---|---|
| TF | count (or count ÷ document length) | **raw count** (not divided by length); optional `sublinear_tf=True` uses 1 + ln(count) |
| IDF | log(N / df) | **ln((1 + N) / (1 + df)) + 1** (`smooth_idf=True`) |
| Effect of smoothing | – | Behaves as if one extra document containing every term exactly once existed, which prevents division by zero |
| Effect of "+1" | Term in all documents gets weight 0 | A term in all documents still gets idf = 1, so it is down-weighted but **not ignored** |
| Normalization | none | each document vector is scaled to **unit Euclidean (L2) length** (`norm='l2'`) |

Because of the L2 normalization, the *absolute* numbers differ from the textbook, and long documents no longer get larger weights simply because they have more words. The *ranking* of terms inside a document is usually similar, though not guaranteed to be identical.

**Weights, not probabilities.** A TF-IDF value is a heuristic *importance score*. It is not the probability of the word, of the class, or of anything else: it is not derived from a probabilistic model of events, the values in a row do not sum to 1 (after L2 normalization their *squares* sum to 1, which is a geometric property of a unit vector), and the numbers depend on the corpus, the smoothing and the normalization used. A value of 0.55 therefore does not mean "55 %"; it only means the term is relatively more important in this document than a term with 0.30.

---

## 6. `fit`, `transform`, `fit_transform`, out-of-vocabulary words and data leakage

**The three methods** (scikit-learn developers, n.d.):

| Method | What it does | Typical use |
|---|---|---|
| `fit(texts)` | **Learns** from the texts: builds the vocabulary (and, for TF-IDF, the document frequencies / IDF values) | training data only |
| `transform(texts)` | **Applies** what was learned: converts texts to vectors with the *existing* columns; it never adds columns and never changes the learned values | training data and any new data |
| `fit_transform(texts)` | `fit` followed by `transform` in one call | training data |

**Out-of-vocabulary (OOV) words.** A word that was not seen during `fit` has no column, so `transform` simply **ignores** it. If *all* words of a new text are unknown, the result is an all-zero row. This is not an error, but it means information is silently lost; a model fed with an all-zero vector can only output its default/prior prediction. Subword tokenizers, character n-grams or stemming reduce the OOV rate.

**Data leakage in vectorization.** *Data leakage* means that information from outside the training data (usually from the test data, or from the target label) influences how the model is built (Kaufman, Rosset, Perlich & Stitelman, 2012). In vectorization, leakage happens when the vectorizer is fitted on *all* texts before the train/test split, because then:

* test-only words enter the vocabulary and the column set;
* IDF values and `min_df`/`max_df`/`max_features` decisions are computed using test documents;
* if the vocabulary is selected using labels (chi-square feature selection, for instance), label information of the test set leaks too, which is usually the most serious case.

**Why evaluation then looks better than reality.** The test set is supposed to simulate *unseen future data*. If the preprocessing has already "seen" it, the model is evaluated under conditions that will not exist in deployment, where genuinely new words and new document frequencies appear. The reported score is optimistic and does not generalize; Kapoor and Narayanan (2023) document leakage as a widespread cause of irreproducible ML results. (For a purely unsupervised vectorizer the numerical inflation is often small, but it breaks the evaluation protocol, and with label-based selection it can be large.)

**Correct order of steps.**

```text
1. Split the data into train / test (and validation, if needed)
2. vectorizer.fit(train_texts)            ← vocabulary and IDF learned from TRAIN only
3. X_train = vectorizer.transform(train_texts)
4. X_test  = vectorizer.transform(test_texts)   ← same fitted vectorizer, no refitting
5. Train the model on X_train, evaluate on X_test
```

During cross-validation the vectorizer has to be re-fitted inside every fold; scikit-learn's `Pipeline` does this automatically when the vectorizer is a pipeline step.

---

## 7. Case study: comparing text representations for Somali news topic classification

**Source.** Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J., Tonja, A. L., Mwase, C., … Stenetorp, P. (2023). *MasakhaNEWS: News Topic Classification for African Languages.* IJCNLP-AACL 2023; arXiv:2304.09972. The study covers 16 African languages, **including Somali**.

**Task.** Single-label *news topic classification*: given a headline plus the news text, predict a topic such as business, entertainment, health, politics, religion, sports or technology.

**Data (Somali part).** Articles were crawled from the BBC Somali website (2,915 articles) and annotated by native speakers. After filtering rare labels, the Somali dataset has **7 topics** and is split into 1,021 training, 148 development and 294 test articles. The inter-annotator agreement (Fleiss' kappa) for Somali is 0.55, the lowest of the 16 languages, which means that Somali labels are somewhat noisier than those of the other languages.

**Representations compared.**

| Family | Representation of the text | Classifiers |
|---|---|---|
| Classical ML | **Bag-of-Words token counts** (scikit-learn `CountVectorizer`) | Naive Bayes, multilayer perceptron (MLP), XGBoost |
| Pre-trained language models | **Contextual embeddings** learned by fine-tuning multilingual Transformers (XLM-R, AfriBERTa, AfroXLMR, LaBSE, RemBERT, …) | Fine-tuned encoders (plus few-shot variants such as PET and SetFit) |

(The authors compare count-based BoW with pre-trained language models; they do not run a separate TF-IDF or n-gram experiment, so the case study illustrates the BoW-versus-contextual-embedding contrast.)

**Results (weighted F1 on the Somali test set, headline + text).**

| Model | Representation | Somali F1 |
|---|---|---:|
| XGBoost | BoW counts | 68.5 |
| MLP | BoW counts | 71.1 |
| Naive Bayes | BoW counts | 74.8 |
| XLM-R-base | contextual embeddings | 73.8 |
| AfriBERTa-large | contextual embeddings | 83.9 |
| AfroXLMR-base | contextual embeddings | 85.2 |
| AfroXLMR-large | contextual embeddings | **86.9** |

Averaged over all 16 languages, the classical models reach 82.8–85.7 F1, whereas the best fine-tuned model (AfroXLMR-large) reaches 92.6 F1. On Somali, the best language model is about 12 F1 points ahead of the best BoW model.

**How the study illustrates "no meaning".** The paper does not use that phrase itself, so the following link is my interpretation of its evidence:

1. **Short text hurts BoW much more.** When only the headline is used (little text, few word matches), the average F1 of the BoW models falls by about 7–10 points compared with headline + text, whereas AfroXLMR changes by only 1.5–2.3 points. A count vector can only match *identical* surface words; a short headline shares few of them with training articles, and BoW has no way to know that two different words describe the same topic. Pre-trained models bring *meaning learned from large unlabeled text*, so they still place a short headline near related articles.
2. **Words are independent columns.** For BoW, "kubbad" (ball) and "ciyaar" (game) are two unrelated columns, even though both signal sport. A model with little training data (here only about 1,000 Somali training articles) cannot learn every such relation from labels alone.
3. **Morphology fragments the vocabulary (hypothesis).** Somali attaches suffixes such as the definite article (e.g., *buug* "book" → *buugga* "the book"), so one concept appears under several surface forms, each a separate column in a count vector. This probably contributes to the weak Somali BoW scores, but the paper does not test it, so it should be read as a plausible explanation, not a proven result.
4. **A fair nuance.** BoW is not useless: it is cheap (the paper lists their size as under 20K), and it is competitive with some large models when the language was not seen in pre-training (for example French and chiShona in the paper). In Somali, Naive Bayes (74.8) even slightly beats XLM-R-base (73.8). This agrees with the general view that BoW and TF-IDF remain strong baselines, but that meaning-aware representations win when semantics or limited text matter.

*Further reading in the same language:* Mohamed et al. (2025) build a Somali-specific Transformer (SomBERTa) for fake-news and toxic-content detection and report that it beats multilingual comparators, which supports the same direction of travel for Somali NLP.

---

## References

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J. O., Tonja, A. L., Mwase, C., … Stenetorp, P. (2023). MasakhaNEWS: News topic classification for African languages. *Proceedings of IJCNLP-AACL 2023*. arXiv:2304.09972.

Harris, Z. S. (1954). Distributional structure. *Word, 10*(2–3), 146–162.

Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects.* Academic Press.

Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. *Proceedings of ECML-98*, 137–142.

Jurafsky, D., & Martin, J. H. (2025). *Speech and Language Processing* (3rd ed., draft).

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15.

Luhn, H. P. (1957). A statistical approach to mechanized encoding and searching of literary information. *IBM Journal of Research and Development, 1*(4), 309–317.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.

Mohamed, M. A., Ahmed, S. D., Isse, Y. A., Mohamed, H. M., Hassan, F. M., & Assowe, H. A. (2025). Detection of Somali-written fake news and toxic messages on the social media using transformer-based language models. arXiv:2503.18117.

Nothman, J., Qin, H., & Yurchak, R. (2018). Stop word lists in free open-source software packages. *Proceedings of the Workshop for NLP Open Source Software (NLP-OSS)*, 7–12.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3), 130–137.

Robertson, S. (2004). Understanding inverse document frequency: On theoretical arguments for IDF. *Journal of Documentation, 60*(5), 503–520.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.

Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM, 18*(11), 613–620.

scikit-learn developers. (n.d.). *Text feature extraction* (user guide) and API documentation for `CountVectorizer`, `TfidfVectorizer` and `TfidfTransformer`.

SciPy developers. (n.d.). *scipy.sparse.csr_matrix* documentation.

Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21.

Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. *Proceedings of ACL 2012 (Short Papers)*, 90–94.

Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort.* Addison-Wesley.

### Reference links (online sources)

1. MasakhaNEWS (case study): https://arxiv.org/abs/2304.09972 (PDF: https://arxiv.org/pdf/2304.09972)
2. SomBERTa, Somali fake news and toxic content: https://arxiv.org/abs/2503.18117
3. Speech and Language Processing (Jurafsky & Martin): https://web.stanford.edu/~jurafsky/slp3/
4. Introduction to Information Retrieval https://www.cambridge.org/9780521865715
5. scikit-learn, Text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html
6. SciPy, `csr_array`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html

---

# Part B – Practical

**Files:** `vectorization_practice.py` (script) and `output.txt` (its full printed output, produced with `python vectorization_practice.py > output.txt`). All numbers below are copied from that output (tables were generated by code from the same data, not typed by hand).

**Data used.**

| ID | Text |
|---|---|
| D1 | Win a FREE phone now! Reply WIN to claim. |
| D2 | Free entry: win cash prizes today. Call now. |
| D3 | Are we still meeting for lunch today? |
| D4 | Please send me the lecture notes before class. |
| D5 | The lecture was not good, but the notes were great. |
| N1 | Claim your free prize now |
| N2 | Is the lunch meeting moved to tomorrow? |
| R1–R3 | the food was tasty / delicious / terrible |

## Summary of the key numbers

| Measure | Result (from `output.txt`) |
|---|---|
| Vocabulary size, D1–D5, default `CountVectorizer` | **32** columns (matrix shape 5 × 32) |
| Total tokens in D1–D5 (vectorizer's tokenization) | **41** |
| Zero cells in the count matrix | **121 of 160 (75.6 %)** |
| Vocabulary with `min_df=2` / `max_features=5` | **7** / **5** |
| Vocabulary with n-grams (1,1) / (1,2) / (1,3) | **32 / 67 / 98** |
| Vocabulary fitted on D1–D4 only vs. D1–D4 + N1 + N2 | **26 vs 31** (5 leaked words) |
| Cosine similarity R1–R2 vs. R1–R3 | **0.75 vs 0.75** (identical) |

---

## Task 1 – One-hot encoding

I wrote `one_hot(word, vocabulary)` in plain Python. With the vocabulary `[cash, class, free, lunch, notes, win]` the output was:

| Word | One-hot vector |
|---|---|
| free | `[0, 0, 1, 0, 0, 0]` |
| cash | `[1, 0, 0, 0, 0, 0]` |
| win | `[0, 0, 0, 0, 0, 1]` |

**Short answer.** `free cash` produces 2 vectors and `win free cash` produces 3, because a one-hot encoder creates one vector per word. The number of vectors therefore depends on the sentence length, but a classifier needs exactly one fixed-length vector per document. This is why one-hot encoding of words is not enough and why Bag of Words, which combines the word vectors into one vector per document, is needed.

---

## Task 2 – Bag of Words

The vocabulary has 32 words and the matrix has shape **(5, 32)**. The document-term matrix (one row per document, one column per word, in the alphabetical order used by `CountVectorizer`):

*(Scroll horizontally if your viewer cuts the table.)*

| Doc | are | before | but | call | cash | claim | class | entry | for | free | good | great | lecture | lunch | me | meeting | not | notes | now | phone | please | prizes | reply | send | still | the | to | today | was | we | were | win |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **D1** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| **D2** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **D3** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **D4** | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| **D5** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 |


**Short answer.** The only cell in D1 with the value 2 is **`win`**, because D1 contains "Win" and "WIN"; `CountVectorizer` lowercases the text, so both forms are counted in the same column. The vectors for `win free phone` and `phone free win` are **identical** (`True` in the output), because Bag of Words ignores word order and only records which words occur and how often.

---

## Task 3 – Tokens vs vocabulary size

The vectorizer's own tokenization (words of two or more letters) gives 8, 8, 7, 8 and 10 tokens for D1–D5, so **41 tokens in total**, while the vocabulary size is **32**. The words that appear in more than one document are:

| Word | df | Documents |
|---|---:|---|
| free | 2 | D1, D2 |
| now | 2 | D1, D2 |
| win | 2 | D1, D2 |
| today | 2 | D2, D3 |
| lecture | 2 | D4, D5 |
| notes | 2 | D4, D5 |
| the | 2 | D4, D5 |

---

## Task 4 – Preprocessing inside the vectorizer

Fitted on D1 only:

| Setting | Vocabulary |
|---|---|
| Default | `claim, free, now, phone, reply, to, win` |
| `lowercase=False` | `FREE, Reply, WIN, Win, claim, now, phone, to` |
| `stop_words="english"` | `claim, free, phone, reply, win` |

**Short answer.** The one-letter word **`a`** is dropped by the default token pattern (it only keeps tokens of two or more characters), and the **`!`** is ignored because punctuation is not part of a token. With the default settings **`Win`** and **`WIN`** merge into one word `win` (count 2); with `lowercase=False` they stay as two separate columns (`Win`, `WIN`) and even `FREE` stays separate from any `free`. The stop-word list removed **`now`** and **`to`** from D1. When fitted on D1–D5, `stop_words="english"` **did remove `not`** (it removed 15 words: *are, before, but, call, for, me, not, now, please, still, the, to, was, we, were*). For D5 this is dangerous: "The lecture was not good, but the notes were great" becomes `good, great, lecture, notes`, so the negation disappears and the review looks purely positive.

---

## Task 5 – Limiting the vocabulary (D1–D5)

| Setting | Size | Vocabulary |
|---|---:|---|
| Full (default) | 32 | all words |
| `min_df=2` | 7 | `free, lecture, notes, now, the, today, win` |
| `max_features=5` | 5 | `free, lecture, now, the, win` |

**Short answer.** Both settings kept **frequent, repeated words** (`free`, `win`, `now`, `the`, `lecture`) and threw away everything that appears only once. With `min_df=2` the rare words `cash`, `prizes`, `claim`, `phone`, `reply` and `entry`, which are typical spam clues, disappeared, and the very common word `the` survived. `max_features=5` ranks by total count (`the` = 3, `win` = 3, then several words with 2); the five-word cut-off shows that ties are broken arbitrarily, so `notes` and `today` (also count 2) were dropped without a meaningful reason.

---

## Task 6 – N-grams

Fitted on D5 only:

| Setting | Vocabulary |
|---|---|
| `ngram_range=(1, 1)` | `but, good, great, lecture, not, notes, the, was, were` (9) |
| `ngram_range=(1, 2)` | the 9 unigrams plus `but the, good but, lecture was, not good, notes were, the lecture, the notes, was not, were great` (18) |

Vocabulary size on D1–D5:

| `ngram_range` | Vocabulary size | Growth vs. unigrams |
|---|---:|---:|
| (1, 1) | 32 | 1.0 × |
| (1, 2) | 67 | 2.1 × |
| (1, 3) | 98 | 3.1 × |

**Short answer.** The bigram **`not good`** keeps the reviewer's opinion, because it ties the negation to the adjective that unigrams treat separately. Adding bigrams roughly doubled the vocabulary (32 → 67) and adding trigrams tripled it (→ 98), even on only five very short documents; each document of *L* words contributes about *L − n + 1* new, mostly unique n-grams per order, so on real data the growth is much larger, and each n-gram is rarer.

---

## Task 7 – TF-IDF

The full TF-IDF matrix (5 × 32, rounded to 2 decimals) is printed in section 7 of `output.txt`. Count versus TF-IDF weight for every word in **D2** ("Free entry: win cash prizes today. Call now."):

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


For **D1** ("Win a FREE phone now! Reply WIN to claim.") the output was:

| Word | Count | TF-IDF weight |
|---|---:|---:|
| claim | 1 | 0.356 |
| free | 1 | 0.287 |
| now | 1 | 0.287 |
| phone | 1 | 0.356 |
| reply | 1 | 0.356 |
| to | 1 | 0.356 |
| win | 2 | 0.574 |


**Short answer.** In D2 every word occurs once, so the differences come only from IDF: `call`, `cash`, `entry` and `prizes` occur in **one** document (D2 only), giving them the larger IDF (2.099), while `free`, `now`, `today` and `win` occur in **two** documents and get a smaller IDF (1.693). That is why the first group has the weight 0.389 and the second 0.314. In D1, `win` has the highest weight (**0.574**) for two reasons that add up: it occurs twice, so its TF is twice as large (raw count 2 × IDF 1.693 = 3.39 before normalization, versus 2.10 for a once-occurring word that appears in only one document), and `win` is also not found in the other documents D3–D5.

---

## Task 8 – TF-IDF by hand for D2 (textbook formula)

The vectorizer tokenizes D2 into 8 words: `free, entry, win, cash, prizes, today, call, now`. N = 5 documents. The df values come from the count matrix (`cash` appears only in D2; `today` appears in D2 and D3).

| Quantity | `cash` | `today` |
|---|---|---|
| count in D2 | 1 | 1 |
| df | 1 | 2 |
| TF = count / 8 | 1/8 = 0.125 | 1/8 = 0.125 |
| IDF = ln(N / df) | ln(5/1) = **1.6094** | ln(5/2) = **0.9163** |
| TF-IDF = TF × IDF | 0.125 × 1.6094 = **0.2012** | 0.125 × 0.9163 = **0.1145** |

**Short answer.** The textbook values are 0.2012 for `cash` and 0.1145 for `today`, so `cash` ranks higher than `today`; scikit-learn's values from Task 7 (0.389 for `cash`, 0.314 for `today`) give **the same ranking**. The numbers themselves differ, because scikit-learn uses the raw count (not count/8), the smoothed IDF ln((1+N)/(1+df)) + 1, and then L2-normalizes the document vector. The reason for the shared ranking is that both methods give the rarer word (`cash`, df = 1) the larger IDF.

---

## Task 9 – Sparse matrix

For the count matrix of Task 2 (`csr_matrix`): total cells = 5 × 32 = **160**, non-zero cells = **39**, zero cells = **121**, which is **75.6 %** zeros. Only the 39 non-zero values (plus their column indices) need to be stored. With only five short documents the matrix is already three-quarters zero; for a real corpus with a vocabulary of thousands of words the share of zeros is far higher.

---

## Task 10 – Fit and transform (fit on D1–D4)

The vocabulary learned from D1–D4 has **26 words**. Transforming the new texts gave:

| Text | Tokens | Known words (1 each) | Out-of-vocabulary words |
|---|---|---|---|
| N1 "Claim your free prize now" | claim, your, free, prize, now | claim, free, now (sum of vector = 3) | **your, prize** |
| N2 "Is the lunch meeting moved to tomorrow?" | is, the, lunch, meeting, moved, to, tomorrow | the, lunch, meeting, to (sum = 4) | **is, moved, tomorrow** |

**Short answer.** The unknown words are `your` and `prize` in N1 and `is`, `moved` and `tomorrow` in N2; `transform` has no column for them and silently ignores them. `prize` is unknown because the training text D2 contains the different surface form **`prizes`**: the vectorizer compares exact strings, so singular and plural are two unrelated words. **Stemming or lemmatization**, the normalization step from Assignment 2, would map both `prizes` and `prize` to the same base form (for example the stem *prize*/*priz*), so that N1's `prize` would match D2's `prizes`.

---

## Task 11 – Data leakage

| Vectorizer fitted on | Size | Vocabulary |
|---|---:|---|
| D1–D4 only (correct) | 26 | `are, before, call, cash, claim, class, entry, for, free, lecture, lunch, me, meeting, notes, now, phone, please, prizes, reply, send, still, the, to, today, we, win` |
| D1–D4 + N1 + N2 (leakage) | 31 | the 26 words above plus `is, moved, prize, tomorrow, your` |

**Short answer.** The words that leaked from the test set into the vocabulary are **`is`, `moved`, `prize`, `tomorrow` and `your`** (5 words). They got a column although they never occur in the training data, which means that the vocabulary (and, with TF-IDF, the IDF values) was influenced by the test set. The right order is: split the data, `fit` the vectorizer on the training text only, then `transform` both training and test text with that fitted vectorizer.

---

## Task 12 – No meaning

Vocabulary: `delicious, food, tasty, terrible, the, was`.

| Review | `delicious` | `food` | `tasty` | `terrible` | `the` | `was` |
|---|---:|---:|---:|---:|---:|---:|
| R1 "the food was tasty" | 0 | 1 | 1 | 0 | 1 | 1 |
| R2 "the food was delicious" | 1 | 1 | 0 | 0 | 1 | 1 |
| R3 "the food was terrible" | 0 | 1 | 0 | 1 | 1 | 1 |

Cosine similarity matrix (from `sklearn.metrics.pairwise.cosine_similarity`):

| | R1 | R2 | R3 |
|---|---:|---:|---:|
| **R1** | 1.00 | 0.75 | 0.75 |
| **R2** | 0.75 | 1.00 | 0.75 |
| **R3** | 0.75 | 0.75 | 1.00 |

**Short answer.** **No**: R1 (`tasty`) is exactly as similar to R2 (`delicious`, 0.75) as to R3 (`terrible`, 0.75), although a person sees that *tasty* and *delicious* are near-synonyms while *terrible* is the opposite. Each review shares the three words `food`, `the` and `was` with the others (dot product 3, vector length 2 each, so 3/4 = 0.75), and the opinion words `tasty`, `delicious` and `terrible` occupy three different columns, so counts cannot see that two of them mean almost the same. This is the main limitation of Bag of Words and TF-IDF; word embeddings address it by giving similar words similar vectors.

---

## Final paragraph – which settings for a spam detector?

For a spam detector I would use **lowercased unigrams plus a small number of bigrams, weighted with TF-IDF (compared against plain counts on a validation set), with a vocabulary cap that does not remove rare spam words, and fitted on the training messages only.** My outputs support each choice. TF-IDF gives more weight to distinctive words: in D1 the repeated spam word `win` received the largest weight (0.574), and in D2 the single-document words `cash`, `prizes`, `call` and `entry` (0.389) outweighed shared words such as `now` (0.314), which is the behavior a spam filter wants. I would **avoid scikit-learn's generic English stop-word list**, because it removed `now`, `to` and `call` from my spam messages and even `not` from D5, which are cues or negations; if I remove stop words I would use a custom list. I would also **avoid a strict `min_df=2` on a small corpus**, because it dropped the typical spam words `cash`, `prizes`, `claim` and `phone` and kept `the`; on a large corpus a small `min_df` (for example 2 or 3) is fine to remove typos. Bigrams such as "free entry" or "claim now" can be useful, but they roughly doubled the vocabulary (32 → 67) and tripled it with trigrams (→ 98), so I would use `ngram_range=(1, 2)` only together with `min_df` or `max_features`, and avoid trigrams. I would **avoid `lowercase=False`**, since it split `Win` and `WIN` into different columns, though a separate "share of capital letters" feature could keep the shouting signal, and I would apply **stemming or lemmatization** so that `prizes` and `prize` share a column (Task 10). Finally, I would **avoid fitting the vectorizer on the test messages**, because Task 11 showed that five test-only words entered the vocabulary; the split must come first, then `fit` on training data, then `transform` both parts. All of this stays a Bag-of-Words system, so it will still fail on paraphrases that avoid the known spam words (Task 12); embeddings or a language model would be the next step.
