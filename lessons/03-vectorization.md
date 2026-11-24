# Vectorization

## Goal

Learn how text becomes the numbers a Machine Learning model expects: one-hot encoding, Bag of Words, the document-term matrix, vocabulary, n-grams, TF-IDF, fitting versus transforming, data leakage, and the main limitation of these methods: they do not capture meaning. The teaching script is `code/vectorization.py`.

## 1. What is text vectorization?

**Text vectorization** means turning a document into numbers. Models such as Logistic Regression accept numerical features, but a message arrives as text, and one message may have four words while another has twenty. We need to describe every document with the **same number of features**.

The idea: choose a **vocabulary**, give each vocabulary word one **position**, and describe every document using those same positions.

The simplest version is **one-hot encoding** of single words: a word's vector has a `1` in its own position and `0` everywhere else.

**Example:** vocabulary `[class, free, notes, prize]` (real output)

| Word | `class` | `free` | `notes` | `prize` |
|---|---:|---:|---:|---:|
| `free` | 0 | 1 | 0 | 0 |
| `prize` | 0 | 0 | 0 | 1 |

One-hot encoding turns a sentence into a **list** of vectors, one per word, so longer sentences give longer lists. Classical models need **one** vector per document. Bag of Words solves this.

## 2. Bag of Words and the document-term matrix

**Bag of Words** represents a document by counting how many times each vocabulary word appears in it. It is called a *bag* because it ignores word order. When we stack the vectors of many documents, we get a **document-term matrix**: one **row** per document and one **column** per vocabulary word.

Scikit-learn's **`CountVectorizer`** learns the vocabulary when it is **fitted** and returns this matrix. It lists the columns in alphabetical order.

**Example:** four teaching documents (real output, shape **(4, 15)**: 4 documents, 15 vocabulary words)

| Document | cash | claim | class | free | in | notes | now | please | prize | see | send | the | today | win | you |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `win free cash now` | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `free prize claim now` | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `see you in class today` | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| `please send the class notes` | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |

Find the `free` column: it is `1` in the first two messages and `0` in the last two. The number of documents decides the rows; the vocabulary decides the columns.

**Word order is lost.** With this vocabulary, `free prize` and `prize free` both get a `1` under `free` and `prize` and `0` everywhere else: identical vectors. For spam detection this is often fine, because words like `prize` and `cash` are strong clues on their own. It is a problem for phrases like `not good`, which n-grams (Section 4) help with.

**Most cells are zero.** Each document uses only a few vocabulary words, so most of the matrix is zero. In the table above, **42 of the 60 cells (70%)** are zero. A matrix like this is called **sparse**, and scikit-learn stores only its non-zero cells to save memory.

> **Remember:** documents become rows, vocabulary words become columns, and the cells hold the numbers.

## 3. The vocabulary: size, limits and preprocessing

A **token** is one word occurrence; the **vocabulary** is the set of unique words the vectorizer keeps. **Vocabulary size** is the number of columns.

**Example:** the four documents above (real output)

| Measure | Value |
|---|---:|
| Tokens (every word occurrence) | 18 |
| Vocabulary size (unique words = columns) | 15 |

`free`, `now` and `class` appear twice each, so they count twice as tokens but once in the vocabulary.

**Limiting the vocabulary.** Real datasets have tens of thousands of words. Two settings keep the vocabulary smaller: `min_df` keeps only words found in at least that many documents, and `max_features` keeps only the most frequent words.

**Example:** (real output)

| Setting | Vocabulary | Size |
|---|---|---:|
| Default | `cash, claim, class, free, in, notes, now, please, prize, see, send, the, today, win, you` | 15 |
| `min_df=2` | `class, free, now` | 3 |
| `max_features=5` | `cash, claim, class, free, now` | 5 |

**The vectorizer does its own preprocessing.** By default `CountVectorizer` lowercases the text, drops punctuation and one-letter words, and it can remove stopwords if we ask. These settings are preprocessing decisions, just like in Lesson 2.

**Example:** the message `FREE Prize! Call the office NOW` (real output)

| Setting | Vocabulary |
|---|---|
| Default | `call`, `free`, `now`, `office`, `prize`, `the` |
| `lowercase=False` | `Call`, `FREE`, `NOW`, `Prize`, `office`, `the` |
| `stop_words="english"` | `free`, `office`, `prize` |

Scikit-learn's English stopword list removed not only `the` but also `call` and `now`, words that could help a spam detector. Always check what a stopword list removes.

## 4. N-grams

An **n-gram** is a sequence of *n* neighboring words. A **unigram** is one word and a **bigram** is two neighboring words. N-grams keep a little of the word order that Bag of Words loses.

**Example:** `not a good movie` (real output)

| Setting | Vocabulary |
|---|---|
| `ngram_range=(1, 1)` (unigrams) | `good`, `movie`, `not` |
| `ngram_range=(1, 2)` (unigrams and bigrams) | `good`, `good movie`, `movie`, `not`, `not good` |

Because the vectorizer drops the one-letter word `a`, it builds the bigram `not good`, the phrase a sentiment model needs. For spam, a bigram like `free prize` can be a stronger clue than either word alone. The cost: every new phrase adds a column, so the vocabulary grows quickly.

## 5. TF-IDF: weighting words by how distinctive they are

Counts tell us how often a word appears in a document, but not whether it appears in **every** document. **TF-IDF** (Term Frequency–Inverse Document Frequency) gives higher weights to words that appear in **fewer** documents.

- **TF:** how often the word appears in this document.
- **IDF:** how rare the word is across all documents.
- **TF-IDF:** TF × IDF.

```text
TF     = (times the word appears in the document) / (words in the document)
IDF    = log(N / df)      N = number of documents, df = documents containing the word
TF-IDF = TF × IDF
```

**Example:** `win` and `free` in `win free cash now`, with N = 4 documents (real output)

| Word | df | TF | IDF | TF-IDF |
|---|---:|---:|---:|---:|
| `win` | 1 | 1 / 4 = 0.25 | log(4) = 1.386 | **0.347** |
| `free` | 2 | 1 / 4 = 0.25 | log(2) = 0.693 | **0.173** |

Both words appear once, so TF is the same. `win` gets the higher weight because it appears in fewer documents.

**Example:** the same document with scikit-learn's `TfidfVectorizer` (real output)

| Word | Count | TF-IDF weight |
|---|---:|---:|
| `win` | 1 | 0.555 |
| `cash` | 1 | 0.555 |
| `free` | 1 | 0.438 |
| `now` | 1 | 0.438 |

Scikit-learn adjusts the textbook formula slightly, so its numbers differ. In this example the ranking is the same: the rarer words `win` and `cash` get higher weights than `free` and `now`.

| | CountVectorizer | TfidfVectorizer |
|---|---|---|
| Cell value | How many times a word appears | A weight: higher for rarer words |
| Typical values | `0`, `1`, `2` | `0.00`, `0.44`, `0.56` |
| Same vocabulary and shape? | Yes | Yes |

TF-IDF values are **weights, not probabilities**: `0.555` does not mean a 55.5% chance of anything. And TF-IDF does not always beat counts; compare both on the same data.

## 6. Fit, transform, data leakage and new words

- **`fit`:** learn the vocabulary (and, for TF-IDF, how many documents contain each word) from **training** text.
- **`transform`:** turn text into numbers using the vocabulary already learned. It never adds columns.
- **`fit_transform`:** both at once, used on the training text.

**Example:** fit on `free prize` and `class notes`, vocabulary `[class, free, notes, prize]` (real output)

| New text | Vector | What happened |
|---|---|---|
| `free notes` | `[0, 1, 1, 0]` | Both words known |
| `free meeting` | `[0, 1, 0, 0]` | `meeting` is **out-of-vocabulary**: no column, ignored |
| `meeting tomorrow` | `[0, 0, 0, 0]` | Every word unknown: an all-zero row |

**Data leakage** happens when test data influences training. If we fit the vectorizer on **all** text before splitting, test words enter the vocabulary.

**Example:** training text `free prize`, `class notes`; test text `free meeting` (real output)

| Fitted on | Vocabulary |
|---|---|
| Training text only (correct) | `class, free, notes, prize` |
| Training + test text (leakage) | `class, free, meeting, notes, prize` |

`meeting` got a column even though it only appears in the test set. The correct order: **split the data → fit on training text → transform both training and test text with the same fitted vectorizer.** The teaching script fits on its tiny corpus only so we can read the matrix; a real project must split first.

## 7. The main limitation: no meaning

Bag of Words and TF-IDF only match **identical words**. Words with almost the same meaning get separate, unrelated columns.

**Example:** two reviews that mean almost the same thing (real output)

| Review | `good` | `great` | `movie` | `the` | `was` |
|---|---:|---:|---:|---:|---:|
| `the movie was good` | 1 | 0 | 1 | 1 | 1 |
| `the movie was great` | 0 | 1 | 1 | 1 | 1 |

The words that carry the opinion, `good` and `great`, do not overlap at all. To these methods, `good` is as different from `great` as it is from `terrible`.

| | Bag of Words / TF-IDF |
|---|---|
| **Strengths** | Simple, fast, easy to explain, strong baselines for classification and search |
| **Weaknesses** | No word meaning, word order lost (except short n-grams), large sparse matrices, unknown words ignored |

**Word embeddings** such as Word2Vec (Lesson 5) address the meaning problem: they give similar words similar vectors.

## 8. Practical: the teaching script

The code for this lesson lives in `code/vectorization.py`, our teaching script. Run `python code/vectorization.py` from the repository root.

The script follows this lesson in twelve numbered steps: one-hot encoding, Bag of Words, tokens versus vocabulary size, preprocessing inside the vectorizer, limiting the vocabulary, n-grams, TF-IDF, TF-IDF by hand, sparse matrices, fitting and transforming new text, data leakage, and the no-meaning limitation. Each step has a short comment explaining what to look for and prints its result under a heading. It trains no model; that comes in the classification lessons.

## Lesson recap

- **Vectorization** turns text into numbers with the same positions for every document; **one-hot encoding** does this for single words.
- **Bag of Words** counts words per document and loses word order; stacked documents form a sparse **document-term matrix**.
- **Vocabulary size** is the number of columns; `min_df` and `max_features` limit it, and the vectorizer's own preprocessing (lowercasing, stopwords) changes it.
- **N-grams** keep short word sequences such as `not good`, at the cost of more columns.
- **TF-IDF** = TF × IDF gives higher weights to rarer words; its values are weights, not probabilities.
- **Fit** on training text only; **transform** reuses the vocabulary, ignores unknown words, and avoids data leakage.
- These methods **do not capture meaning**; word embeddings (Lesson 5) address this.